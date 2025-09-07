# AI-Image-Caption-Recommender-Python-FastAPI-Streamlit-PyTorch-Hugging-Face

An intelligent system that generates and ranks image captions using state-of-the-art AI models. Upload an image, provide keywords, and receive AI-generated captions ranked by relevance to your image content.

🌟 Features
Dual Caption Generation Methods:
Keyword-based: Generate creative captions using GPT-2 based on your keywords
Predefined List: Rank your custom captions by relevance to the image

Multiple Processing Modes:
Single image processing with detailed analysis
Batch processing for numerous images
Comparison mode to analyze captions across images


Advanced AI Integration:
CLIP model for image-text similarity scoring
GPT-2 for creative caption generation
Confidence scoring for all recommendations

Professional UI/UX:
Modern, responsive interface with dark mode
Interactive visualizations of confidence scores
Progress tracking for batch operations
Export functionality for results

RESTful API:
Full FastAPI backend for programmatic access
Batch processing endpoints
Health check and monitoring endpoints

🛠️ Technology Stack
Core Technologies
Backend: Python 3.9+, FastAPI, PyTorch
Frontend: Streamlit, Plotly, HTML/CSS

AI Models:
OpenAI CLIP (ViT-Base-Patch32)
GPT-2 for text generation
Deployment: Docker-ready, uvicorn server

Key Libraries:
transformers (Hugging Face) - Pre-trained models
PIL - Image processing
numpy - Numerical operations
plotly - Data visualization
pydantic - Data validation

🚀 Installation & Setup
Prerequisites
Python 3.9 or higher
pip package manager
4GB+ RAM (for model loading)

Download Models (First Run)
Models will be automatically downloaded on first run:

CLIP model (~350MB)
GPT-2 model (~550MB)

💻 Usage
Web Application
Start the Streamlit app:streamlit run app.py

[How to Use:
Upload an Image: Drag & drop or select from sample images
Choose Caption Method:
Keyword-based: Enter keywords (e.g., "nature, landscape, sunset")
Predefined List: Enter candidate captions (one per line)
Adjust Settings:
Number of recommendations (1-10)
Number of candidates to generate (5-20)
Get Recommendations: Click "Get Recommendations"
View Results:
Confidence scores with visual indicators
Interactive charts showing similarity scores
Export results as JSON]


📊 Project Structure
ai-caption-recommender/
├── app.py                     # Streamlit web application
├── api_integration.py         # FastAPI backend
├── caption_recommendation.py  # Core AI recommendation system
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── temp_uploads/             # Temporary file storage (auto-created)



Key Components
caption_recommendation.py:
        ImageCaptionRecommendationSystem class
        CLIP model integration for image-text similarity
        GPT-2 integration for caption generation
        Methods for both keyword-based and predefined caption ranking
app.py:
      Streamlit frontend implementation
      Multi-mode processing (single, batch, comparison)
      Interactive visualizations
        Dark mode support
api_integration.py:
      FastAPI RESTful endpoints
      File upload handling
      Background task processing
      Comprehensive error handling


Model Selection
You can change the CLIP model by modifying:
model_name in ImageCaptionRecommendationSystem
Supported models: openai/clip-vit-base-patch32, openai/clip-vit-large-patch14


📈 Performance
Processing Time: ~2-5 seconds per image (GPU), ~5-15 seconds (CPU)
Memory Usage: ~2GB RAM (models loaded once)
Supported Formats: JPG, PNG, BMP, TIFF, WebP
Batch Processing: Efficient handling of multiple images with progress tracking


🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository:-
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request


Development Guidelines:-
Follow PEP 8 style guidelines
Add docstrings to all new functions
Write tests for new features
Update documentation as needed
