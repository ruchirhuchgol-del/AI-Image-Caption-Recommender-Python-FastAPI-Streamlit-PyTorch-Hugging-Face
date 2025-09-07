"""
AI Image Caption Recommendation System


""" 

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer

# Suppress transformer warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Enum for confidence levels based on similarity scores."""
    EXCELLENT = ("🔥 EXCELLENT", 0.8)
    GOOD = ("✅ GOOD", 0.6)
    FAIR = ("⚠️  FAIR", 0.4)
    POOR = ("❌ POOR", 0.0)
    
    def __init__(self, label: str, threshold: float):
        self.label = label
        self.threshold = threshold
    
    @classmethod
    def get_confidence(cls, score: float) -> 'ConfidenceLevel':
        """Get confidence level based on similarity score."""
        for level in [cls.EXCELLENT, cls.GOOD, cls.FAIR, cls.POOR]:
            if score >= level.threshold:
                return level
        return cls.POOR


@dataclass
class CaptionRecommendation:
    """Data class for caption recommendations."""
    caption: str
    similarity_score: float
    confidence_level: ConfidenceLevel
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise ValueError("Caption must be a non-empty string")
        if not isinstance(self.similarity_score, (int, float)):
            raise ValueError("Similarity score must be numeric")
        if not -1.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between -1.0 and 1.0")
        
        self.confidence_level = ConfidenceLevel.get_confidence(self.similarity_score)


@dataclass
class ModelConfig:
    """Configuration for the recommendation system models."""
    clip_model_name: str = "openai/clip-vit-base-patch32"
    caption_model_name: str = "gpt2"
    max_text_length: int = 77  # CLIP's max token length
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_text_length <= 0:
            raise ValueError("max_text_length must be positive")


@dataclass
class GenerationConfig:
    """Configuration for caption generation."""
    max_length: int = 50
    num_return_sequences: int = 1
    no_repeat_ngram_size: int = 2
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.7
    
    def __post_init__(self):
        """Validate generation parameters."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")


class ImageProcessor:
    """Handles image loading and preprocessing."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    @classmethod
    def load_and_validate(cls, image_path: Union[str, Path]) -> Image.Image:
        """Load and validate image from path."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats: {', '.join(cls.SUPPORTED_FORMATS)}"
            )
        
        try:
            image = Image.open(image_path)
            # Convert to RGB to ensure compatibility
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}") from e


class BaseModel(ABC):
    """Abstract base class for models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._model = None
        self._processor = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._processor is not None


class CLIPEmbedder(BaseModel):
    """Handles CLIP model operations for embeddings."""
    
    def load_model(self) -> None:
        """Load CLIP model and processor."""
        try:
            logger.info(f"Loading CLIP model: {self.config.clip_model_name}")
            self._model = CLIPModel.from_pretrained(self.config.clip_model_name)
            self._processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
            
            self._model.to(self.device)
            self._model.eval()
            
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}") from e
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to embeddings."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                # Normalize for better cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu()
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise RuntimeError(f"Image encoding failed: {e}") from e
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings")
        
        try:
            inputs = self._processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                # Normalize for better cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise RuntimeError(f"Text encoding failed: {e}") from e


class CaptionGenerator(BaseModel):
    """Handles caption generation using GPT-2."""
    
    def __init__(self, config: ModelConfig, generation_config: GenerationConfig):
        super().__init__(config)
        self.generation_config = generation_config
        self._tokenizer = None
    
    def load_model(self) -> None:
        """Load GPT-2 model and tokenizer."""
        try:
            logger.info(f"Loading caption model: {self.config.caption_model_name}")
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.config.caption_model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.config.caption_model_name)
            
            # Set pad token for GPT-2
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._model.to(self.device)
            self._model.eval()
            
            logger.info("Caption model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load caption model: {e}")
            raise RuntimeError(f"Caption model loading failed: {e}") from e
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._tokenizer is not None
    
    def generate_captions(self, keywords: List[str], num_captions: int = 10) -> List[str]:
        """Generate candidate captions based on keywords."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not keywords or not all(isinstance(kw, str) and kw.strip() for kw in keywords):
            raise ValueError("All keywords must be non-empty strings")
        
        if num_captions <= 0:
            raise ValueError("num_captions must be positive")
        
        try:
            prompt = f"A creative caption for an image with {', '.join(keywords)}:"
            unique_captions = set()
            max_attempts = num_captions * 3  # Prevent infinite loop
            attempts = 0
            
            while len(unique_captions) < num_captions and attempts < max_attempts:
                attempts += 1
                
                inputs = self._tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        inputs['input_ids'],
                        max_length=self.generation_config.max_length,
                        num_return_sequences=self.generation_config.num_return_sequences,
                        no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size,
                        do_sample=self.generation_config.do_sample,
                        top_k=self.generation_config.top_k,
                        top_p=self.generation_config.top_p,
                        temperature=self.generation_config.temperature,
                        pad_token_id=self._tokenizer.eos_token_id,
                        eos_token_id=self._tokenizer.eos_token_id
                    )
                
                caption = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                caption = caption.replace(prompt, "").strip()
                
                # Validate caption quality
                if (caption and len(caption.split()) >= 3 and
                    any(kw.lower() in caption.lower() for kw in keywords)):
                    unique_captions.add(caption)
            
            result = list(unique_captions)[:num_captions]
            if not result:
                logger.warning("No valid captions generated")
            return result
            
        except Exception as e:
            logger.error(f"Error generating captions: {e}")
            raise RuntimeError(f"Caption generation failed: {e}") from e


class SimilarityCalculator:
    """Handles similarity calculations between embeddings."""
    
    @staticmethod
    def cosine_similarity(image_features: torch.Tensor, text_features: torch.Tensor) -> np.ndarray:
        """Calculate cosine similarity between image and text features."""
        if image_features.dim() != 2 or text_features.dim() != 2:
            raise ValueError("Both tensors must be 2D")
        
        if image_features.shape[1] != text_features.shape[1]:
            raise ValueError("Feature dimensions must match")
        
        try:
            # Convert to numpy and calculate dot product (features are normalized)
            image_np = image_features.detach().cpu().numpy()
            text_np = text_features.detach().cpu().numpy()
            
            similarities = np.dot(image_np, text_np.T).flatten()
            similarities = np.clip(similarities, -1.0, 1.0)
            
            return similarities
        except Exception as e:
            raise RuntimeError(f"Similarity calculation failed: {e}") from e


class ImageCaptionRecommendationSystem:
    """
    Production-ready AI Image Caption Recommendation System.
    
    Provides functionality to:
    - Load and process images
    - Generate image embeddings using CLIP
    - Generate candidate captions using GPT-2
    - Rank captions by relevance to the image
    - Return top-n recommendations with confidence scores
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None,
                 generation_config: Optional[GenerationConfig] = None):
        """Initialize the recommendation system."""
        self.model_config = model_config or ModelConfig()
        self.generation_config = generation_config or GenerationConfig()
        
        self._clip_embedder = CLIPEmbedder(self.model_config)
        self._caption_generator = CaptionGenerator(self.model_config, self.generation_config)
        self._similarity_calculator = SimilarityCalculator()
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all models."""
        try:
            self._clip_embedder.load_model()
            self._caption_generator.load_model()
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def rank_predefined_captions(
        self,
        image_path: Union[str, Path],
        candidate_captions: List[str],
        top_n: int = 5
    ) -> List[CaptionRecommendation]:
        """
        Rank predefined captions by their relevance to the image.
        
        Args:
            image_path: Path to the input image
            candidate_captions: List of predefined candidate captions
            top_n: Number of top recommendations to return
            
        Returns:
            List of CaptionRecommendation objects sorted by similarity
        """
        if not candidate_captions:
            raise ValueError("Candidate captions list cannot be empty")
        
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        
        try:
            logger.info(f"Processing image: {image_path}")
            logger.info(f"Ranking {len(candidate_captions)} predefined captions")
            
            # Load and process image
            image = ImageProcessor.load_and_validate(image_path)
            image_features = self._clip_embedder.encode_image(image)
            
            # Generate text embeddings
            text_features = self._clip_embedder.encode_text(candidate_captions)
            
            # Calculate similarities
            similarities = self._similarity_calculator.cosine_similarity(image_features, text_features)
            
            # Create recommendations
            recommendations = []
            for caption, score in zip(candidate_captions, similarities):
                try:
                    rec = CaptionRecommendation(
                        caption=caption,
                        similarity_score=float(score),
                        confidence_level=ConfidenceLevel.get_confidence(float(score))
                    )
                    recommendations.append(rec)
                except ValueError as e:
                    logger.warning(f"Skipping invalid recommendation: {e}")
                    continue
            
            # Sort by similarity score and return top-n
            recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
            result = recommendations[:top_n]
            
            logger.info(f"Generated {len(result)} recommendations")
            return result
            
        except Exception as e:
            logger.error(f"Error in rank_predefined_captions: {e}")
            raise
    
    def recommend_captions(
        self,
        image_path: Union[str, Path],
        keywords: List[str],
        top_n: int = 5,
        num_candidates: int = 10
    ) -> List[CaptionRecommendation]:
        """
        Generate and recommend captions based on keywords and image similarity.
        
        Args:
            image_path: Path to the input image
            keywords: List of keywords to include in captions
            top_n: Number of top recommendations to return
            num_candidates: Number of candidate captions to generate
            
        Returns:
            List of CaptionRecommendation objects sorted by similarity
        """
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        
        try:
            logger.info(f"Processing image: {image_path}")
            logger.info(f"Generating captions with keywords: {keywords}")
            
            # Generate candidate captions
            candidate_captions = self._caption_generator.generate_captions(keywords, num_candidates)
            
            if not candidate_captions:
                logger.warning("No candidate captions generated")
                return []
            
            logger.info(f"Generated {len(candidate_captions)} candidate captions")
            
            # Use the predefined caption ranking method
            return self.rank_predefined_captions(image_path, candidate_captions, top_n)
            
        except Exception as e:
            logger.error(f"Error in recommend_captions: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models."""
        return {
            "clip_model": self.model_config.clip_model_name,
            "caption_model": self.model_config.caption_model_name,
            "device": self.model_config.device,
            "clip_loaded": str(self._clip_embedder.is_loaded),
            "caption_generator_loaded": str(self._caption_generator.is_loaded)
        }


def display_recommendations(recommendations: List[CaptionRecommendation], title: str) -> None:
    """Display recommendations in a formatted way."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print("="*60)
    
    if not recommendations:
        print("❌ No recommendations generated.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec.caption}")
        print(f"    Similarity Score: {rec.similarity_score:.4f}")
        print(f"    Confidence: {rec.similarity_score*100:.2f}% {rec.confidence_level.label}")
        print("-" * 50)


def main():
    """
    Main function demonstrating the usage of ImageCaptionRecommendationSystem.
    """
    try:
        # Initialize configuration
        model_config = ModelConfig()
        generation_config = GenerationConfig()
        
        print("🚀 Initializing AI Image Caption Recommendation System...")
        recommender = ImageCaptionRecommendationSystem(model_config, generation_config)
        
        # Print model information
        model_info = recommender.get_model_info()
        print("\n📋 Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Example usage
        image_path = "sample_image.jpg"  # Replace with actual image path
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"\n⚠️  Sample image '{image_path}' not found.")
            print("Please update the image_path variable with a valid image file.")
            print(f"Supported formats: {', '.join(ImageProcessor.SUPPORTED_FORMATS)}")
            return
        
        # Test keyword-based caption generation
        keywords = ["nature", "landscape", "beautiful"]
        print(f"\n📸 Processing image: {image_path}")
        print(f"🔍 Keywords: {', '.join(keywords)}")
        
        keyword_recommendations = recommender.recommend_captions(
            image_path=image_path,
            keywords=keywords,
            top_n=5,
            num_candidates=15
        )
        
        display_recommendations(keyword_recommendations, "KEYWORD-BASED CAPTION RECOMMENDATIONS")
        
        # Test predefined caption ranking
        predefined_captions = [
            "A breathtaking natural landscape",
            "Beautiful scenery in nature",
            "Trees and mountains in harmony",
            "A refreshing outdoor view",
            "The perfect nature getaway",
            "Serene natural beauty",
            "Landscape photography at its finest",
            "Nature's magnificent display",
            "A peaceful outdoor scene"
        ]
        
        predefined_recommendations = recommender.rank_predefined_captions(
            image_path=image_path,
            candidate_captions=predefined_captions,
            top_n=5
        )
        
        display_recommendations(predefined_recommendations, "PREDEFINED CAPTION RANKING")
        
        print(f"\n✅ Successfully processed recommendations!")
        print(f"📊 Generated {len(keyword_recommendations)} keyword-based and "
              f"{len(predefined_recommendations)} predefined recommendations!")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()