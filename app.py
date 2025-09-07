import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="AI Caption Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def local_css():
    st.markdown("""
    <style>
        :root {
            --primary: #FF6B6B;
            --secondary: #4ECDC4;
            --dark-bg: #1a202c;
            --dark-card: #2d3748;
            --light-bg: #f7fafc;
            --light-card: white;
        }
        
        .main-header {
            font-size: 2.8rem !important;
            font-weight: 800;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .subheader {
            font-size: 1.3rem;
            color: #718096;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .upload-container {
            border: 2px dashed #CBD5E0;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            background: var(--light-bg);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .upload-container:hover {
            border-color: var(--secondary);
            background: #EBF8FF;
        }
        
        .upload-container::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(255,107,107,0.05), rgba(78,205,196,0.05));
            z-index: 0;
        }
        
        .upload-content {
            position: relative;
            z-index: 1;
        }
        
        .result-card {
            background: var(--light-card);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            border-left: 5px solid var(--secondary);
        }
        
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        
        .confidence-bar {
            height: 10px;
            border-radius: 10px;
            background: #E2E8F0;
            margin: 0.8rem 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            animation: fillAnimation 1.5s ease-in-out;
        }
        
        @keyframes fillAnimation {
            0% { width: 0; }
            100% { width: var(--width); }
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .dark-mode {
            background-color: var(--dark-bg);
            color: #e2e8f0;
        }
        
        .dark-mode .result-card {
            background: var(--dark-card);
            color: #e2e8f0;
            border-left-color: var(--secondary);
        }
        
        .dark-mode .upload-container {
            background: var(--dark-card);
            border-color: #4a5568;
        }
        
        .dark-mode .metric-card {
            background: linear-gradient(135deg, #434343 0%, #000000 100%);
        }
        
        .tab-container {
            margin-top: 2rem;
        }
        
        .sample-image {
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .sample-image:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .footer {
            margin-top: 3rem;
            padding: 1.5rem;
            text-align: center;
            color: #718096;
            border-top: 1px solid #e2e8f0;
        }
        
        .dark-mode .footer {
            border-top-color: #4a5568;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .method-toggle {
            display: flex;
            justify-content: center;
            margin: 1rem 0;
        }
        
        .method-toggle button {
            background: none;
            border: 2px solid #CBD5E0;
            padding: 0.5rem 1rem;
            margin: 0 0.5rem;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .method-toggle button.active {
            background: var(--secondary);
            color: white;
            border-color: var(--secondary);
        }
        
        .dark-mode .method-toggle button {
            border-color: #4a5568;
            color: #e2e8f0;
        }
        
        .dark-mode .method-toggle button.active {
            background: var(--secondary);
            border-color: var(--secondary);
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Initialize session state
def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0
    if 'tab' not in st.session_state:
        st.session_state.tab = 'single'
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = {}
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None
    if 'caption_method' not in st.session_state:
        st.session_state.caption_method = 'keyword'
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}

init_session_state()

# Define ImageCaptionRecommendationSystem class
class ImageCaptionRecommendationSystem:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def recommend_captions(self, image_path, keywords, top_n=5, num_candidates=10):
        # Generate candidate captions from keywords
        candidate_captions = []
        for _ in range(num_candidates):
            np.random.shuffle(keywords)
            num_words = np.random.randint(1, min(len(keywords), 5) + 1)
            caption = " ".join(keywords[:num_words])
            candidate_captions.append(caption)
        
        # Remove duplicates
        candidate_captions = list(set(candidate_captions))
        
        # Get top captions using CLIP
        top_captions, top_scores = get_top_captions(
            image_path=image_path,
            candidate_captions=candidate_captions,
            top_n=top_n,
            model=self.model,
            processor=self.processor
        )
        
        return list(zip(top_captions, top_scores))

# Load model with caching
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    try:
        with st.spinner("🚀 Loading AI models..."):
            return ImageCaptionRecommendationSystem(model_name)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Helper functions for IPYNB integration
def load_and_preprocess_image(image_path: str, processor: CLIPProcessor) -> dict:
    """Load and preprocess an image for CLIP model."""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def generate_image_embeddings(inputs: dict, model: CLIPModel) -> torch.Tensor:
    """Generate image embeddings using CLIP model."""
    model.eval()
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

def rank_captions(
    image_features: torch.Tensor, 
    captions: List[str], 
    model: CLIPModel, 
    processor: CLIPProcessor
) -> Tuple[List[str], List[float]]:
    """Rank captions by similarity to image features."""
    # Process text inputs
    text_inputs = processor(
        text=captions, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Generate text features
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity scores
    similarity_scores = torch.matmul(image_features, text_features.T).squeeze(0)
    
    # Sort captions by similarity score (descending)
    sorted_indices = torch.argsort(similarity_scores, descending=True)
    sorted_captions = [captions[i] for i in sorted_indices]
    sorted_scores = similarity_scores[sorted_indices].tolist()
    
    return sorted_captions, sorted_scores

def get_top_captions(
    image_path: str, 
    candidate_captions: List[str], 
    top_n: int = 5,
    model: CLIPModel = None,
    processor: CLIPProcessor = None
) -> Tuple[List[str], List[float]]:
    """Get top-n captions for an image from candidate list."""
    # Load models if not provided
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if processor is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load and preprocess image
    inputs = load_and_preprocess_image(image_path, processor)
    
    # Generate image embeddings
    image_features = generate_image_embeddings(inputs, model)
    
    # Rank captions by similarity
    sorted_captions, sorted_scores = rank_captions(
        image_features, candidate_captions, model, processor
    )
    
    # Return top-n captions and scores
    top_n = min(top_n, len(sorted_captions))
    return sorted_captions[:top_n], sorted_scores[:top_n]

# Function to get sample images
@st.cache_data
def get_sample_images():
    return [
        {"name": "Nature Scene", "path": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80"},
        {"name": "Urban Landscape", "path": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80"},
        {"name": "Food & Drink", "path": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80"}
    ]

# Function to create a gauge chart for confidence
def create_gauge_chart(score):
    fig = go.Figure(go.Indicator())
    mode = "gauge+number+delta",
    value = score * 100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Confidence Score", 'font': {'size': 16}},
    gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Function to create comparison chart
def create_comparison_chart(results):
    if not results:
        return None
        
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.05
    )
    
    # Bar chart
    for img_path, recommendations in results.items():
        fig.add_trace(
            go.Bar(
                x=[r["caption"] for r in recommendations],
                y=[r["confidence"] for r in recommendations],
                name=img_path.split('/')[-1],
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Radar chart
    categories = [r["caption"] for r in list(results.values())[0]]
    for img_path, recommendations in results.items():
        fig.add_trace(
            go.Scatterpolar(
                r=[r["confidence"] for r in recommendations],
                theta=categories,
                fill='toself',
                name=img_path.split('/')[-1]
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title="Caption Comparison",
        height=500,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True
    )
    
    return fig

# Process image function
def process_image(image_path, keywords, top_n, num_candidates, caption_method):
    if not keywords:
        return None
    
    try:
        # Get recommendations based on selected method
        if caption_method == "Keyword-based (GPT-2)":
            recommendations = st.session_state.model.recommend_captions(
                image_path=image_path,
                keywords=keywords,
                top_n=top_n,
                num_candidates=num_candidates
            )
            
            # Store results
            results = [
                {
                    "caption": caption,
                    "score": score,
                    "confidence": score * 100
                }
                for caption, score in recommendations
            ]
        else:
            # Use predefined captions method
            top_captions, top_scores = get_top_captions(
                image_path=image_path,
                candidate_captions=keywords,
                top_n=top_n,
                model=st.session_state.model.model,
                processor=st.session_state.model.processor
            )
            
            # Store results
            results = [
                {
                    "caption": caption,
                    "score": score,
                    "confidence": score * 100
                }
                for caption, score in zip(top_captions, top_scores)
            ]
        
        return results
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Sidebar controls
def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Dark mode toggle
        st.session_state.dark_mode = st.toggle(
            "🌙 Dark Mode", 
            value=st.session_state.dark_mode
        )
        
        st.markdown("---")
        
        # Mode selection
        st.subheader("🔧 Processing Mode")
        st.session_state.tab = st.radio(
            "Select mode:",
            ["Single Image", "Batch Processing", "Comparison Mode"],
            index=0
        )
        
        st.markdown("---")
        
        # Model selection
        model_options = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14"
        ]
        selected_model = st.selectbox(
            "🧠 Model Selection",
            model_options,
            index=0
        )
        
        st.markdown("---")
        
        # Caption method selection
        st.subheader("📝 Caption Method")
        st.session_state.caption_method = st.radio(
            "Select caption generation method:",
            ["Keyword-based (GPT-2)", "Predefined List"],
            index=0
        )
        
        if st.session_state.caption_method == "Keyword-based (GPT-2)":
            # Keywords input
            keywords_input = st.text_input(
                "Enter keywords (comma separated):",
                value="nature, landscape, beautiful",
                help="Enter keywords that you want the caption to include, separated by commas"
            )
            keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
            
            # Number of candidates selector
            num_candidates = st.slider(
                "📝 Number of Candidates to Generate",
                min_value=5,
                max_value=20,
                value=10
            )
        else:
            # Predefined captions input
            predefined_captions = st.text_area(
                "Enter candidate captions (one per line):",
                value="Trees, Travel and Tea!\nA refreshing beverage.\nA moment of indulgence.\nThe perfect thirst quencher.\nYour daily dose of delight.\nTaste the tradition.\nSavor the flavor.\nRefresh and rejuvenate.\nUnwind and enjoy.",
                height=200
            )
            keywords = [cap.strip() for cap in predefined_captions.split('\n') if cap.strip()]
            num_candidates = 0  # Not used for predefined method
        
        st.markdown("---")
        
        # Top N selector
        top_n = st.slider(
            "🎯 Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Export button
        if st.session_state.results or st.session_state.batch_results or st.session_state.comparison_results:
            st.download_button(
                label="📥 Export Results",
                data=json.dumps({
                    "single_results": st.session_state.results,
                    "batch_results": st.session_state.batch_results,
                    "comparison_results": st.session_state.comparison_results
                }, indent=2),
                file_name="caption_recommendations.json",
                mime="application/json"
            )
    
    return selected_model, keywords, top_n, num_candidates

# Main application
def main():
    # Apply dark mode
    if st.session_state.dark_mode:
        st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">AI Caption Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload an image and get AI-powered caption recommendations</p>', unsafe_allow_html=True)
    
    # Get sidebar settings
    selected_model, keywords, top_n, num_candidates = render_sidebar()
    
    # Load model
    if st.session_state.model is None or st.session_state.model.model_name != selected_model:
        st.session_state.model = load_model(selected_model)
    
    # Tab container
    tab_container = st.container()
    
    with tab_container:
        # Single Image Mode
        if st.session_state.tab == "Single Image":
            st.markdown("### 📸 Upload Image")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Drag & drop an image here",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                help="Supported formats: JPG, PNG, BMP, TIFF, WebP"
            )
            
            # Sample images
            st.markdown("### Or try with sample images:")
            sample_images = get_sample_images()
            
            sample_cols = st.columns(len(sample_images))
            for i, img in enumerate(sample_images):
                with sample_cols[i]:
                    st.image(img["path"], caption=img["name"], use_column_width=True)
                    if st.button(f"Use {img['name']}", key=f"sample_{i}"):
                        st.session_state.selected_sample = img
            
            # Process uploaded image or selected sample
            if uploaded_file is not None or st.session_state.selected_sample:
                # Display image
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                else:
                    st.image(st.session_state.selected_sample["path"], 
                            caption=st.session_state.selected_sample["name"], 
                            use_column_width=True)
                
                # Process button
                if st.button("🔍 Get Recommendations", type="primary"):
                    if not keywords:
                        st.error("Please enter at least one keyword or caption")
                    else:
                        start_time = time.time()
                        
                        with st.spinner("🤖 Analyzing image..."):
                            # Save uploaded file temporarily
                            temp_path = None
                            try:
                                if uploaded_file:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                                        f.write(uploaded_file.getbuffer())
                                        temp_path = f.name
                                else:
                                    # Download sample image
                                    response = requests.get(st.session_state.selected_sample["path"])
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                                        f.write(response.content)
                                        temp_path = f.name
                                
                                # Process the image
                                results = process_image(
                                    temp_path, 
                                    keywords, 
                                    top_n, 
                                    num_candidates, 
                                    st.session_state.caption_method
                                )
                                
                                if results:
                                    st.session_state.results = results
                                    st.session_state.processing_time = time.time() - start_time
                                
                            except Exception as e:
                                st.error(f"Error processing image: {str(e)}")
                            finally:
                                # Clean up temporary file
                                if temp_path and os.path.exists(temp_path):
                                    os.unlink(temp_path)
            
            # Display results
            if st.session_state.results:
                st.markdown("### 🎯 Top Recommendations")
                
                # Metrics row
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⏱️ Processing Time</h3>
                        <h2>{st.session_state.processing_time:.2f}s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[1]:
                    method_name = "Keywords" if st.session_state.caption_method == "Keyword-based (GPT-2)" else "Captions"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 {method_name} Used</h3>
                        <h2>{len(keywords)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[2]:
                    avg_confidence = np.mean([r["confidence"] for r in st.session_state.results])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔥 Avg Confidence</h3>
                        <h2>{avg_confidence:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Results visualization
                st.markdown("### 📈 Confidence Visualization")
                
                # Gauge chart for top result
                top_result = st.session_state.results[0]
                fig = create_gauge_chart(top_result["score"])
                st.plotly_chart(fig, use_container_width=True)
                
                # Bar chart for all results
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[r["caption"] for r in st.session_state.results],
                    y=[r["confidence"] for r in st.session_state.results],
                    marker_color='rgba(78, 205, 196, 0.8)',
                    text=[f"{r['confidence']:.1f}%" for r in st.session_state.results],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Caption Confidence Scores",
                    xaxis_title="Captions",
                    yaxis_title="Confidence (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.markdown("### 📝 Detailed Results")
                
                for i, result in enumerate(st.session_state.results, 1):
                    # Confidence level
                    if result["confidence"] >= 80:
                        confidence_level = "🔥 EXCELLENT"
                        color = "#48BB78"
                    elif result["confidence"] >= 60:
                        confidence_level = "✅ GOOD"
                        color = "#4299E1"
                    elif result["confidence"] >= 40:
                        confidence_level = "⚠️ FAIR"
                        color = "#ED8936"
                    else:
                        confidence_level = "❌ POOR"
                        color = "#F56565"
                    
                    # Result card
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>{i}. {result['caption']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.1f}% {confidence_level}</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="--width: {result['confidence']}%; background: {color};"></div>
                        </div>
                        <p><strong>Similarity Score:</strong> {result['score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Batch Processing Mode
        elif st.session_state.tab == "Batch Processing":
            st.markdown("### 📸 Upload Multiple Images")
            
            uploaded_files = st.file_uploader(
                "Drag & drop multiple images here",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                accept_multiple_files=True,
                help="Supported formats: JPG, PNG, BMP, TIFF, WebP"
            )
            
            if uploaded_files:
                # Display images
                st.markdown("### Uploaded Images")
                cols = st.columns(min(4, len(uploaded_files)))
                for i, file in enumerate(uploaded_files):
                    with cols[i % 4]:
                        st.image(file, caption=file.name, use_column_width=True)
                
                # Process button
                if st.button("🔍 Process Batch", type="primary"):
                    if not keywords:
                        st.error("Please enter at least one keyword or caption")
                    else:
                        start_time = time.time()
                        
                        with st.spinner("🤖 Analyzing images..."):
                            # Process each image
                            batch_results = {}
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, file in enumerate(uploaded_files):
                                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
                                
                                # Save uploaded file temporarily
                                temp_path = None
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                                        f.write(file.getbuffer())
                                        temp_path = f.name
                                    
                                    # Process the image
                                    results = process_image(
                                        temp_path, 
                                        keywords, 
                                        top_n, 
                                        num_candidates, 
                                        st.session_state.caption_method
                                    )
                                    
                                    if results:
                                        batch_results[file.name] = results
                                    
                                except Exception as e:
                                    st.error(f"Error processing {file.name}: {str(e)}")
                                finally:
                                    # Clean up temporary file
                                    if temp_path and os.path.exists(temp_path):
                                        os.unlink(temp_path)
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(uploaded_files))
                            
                            processing_time = time.time() - start_time
                            st.session_state.batch_results = batch_results
                            st.session_state.processing_time = processing_time
                            
                            status_text.text(f"✅ Processed {len(uploaded_files)} images in {processing_time:.2f}s")
            
            # Display batch results
            if st.session_state.batch_results:
                st.markdown("### 🎯 Batch Results")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Images Processed", len(st.session_state.batch_results))
                with col2:
                    st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
                with col3:
                    avg_conf = np.mean([
                        np.mean([r["confidence"] for r in results])
                        for results in st.session_state.batch_results.values()
                    ])
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
                # Results visualization
                st.markdown("### 📈 Batch Confidence Visualization")
                
                # Create comparison chart
                fig = create_comparison_chart(st.session_state.batch_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.markdown("### 📝 Detailed Results")
                
                for img_name, results in st.session_state.batch_results.items():
                    with st.expander(f"🖼️ {img_name}"):
                        for i, result in enumerate(results, 1):
                            # Confidence level
                            if result["confidence"] >= 80:
                                confidence_level = "🔥 EXCELLENT"
                                color = "#48BB78"
                            elif result["confidence"] >= 60:
                                confidence_level = "✅ GOOD"
                                color = "#4299E1"
                            elif result["confidence"] >= 40:
                                confidence_level = "⚠️ FAIR"
                                color = "#ED8936"
                            else:
                                confidence_level = "❌ POOR"
                                color = "#F56565"
                            
                            # Result card
                            st.markdown(f"""
                            <div class="result-card">
                                <h3>{i}. {result['caption']}</h3>
                                <p><strong>Confidence:</strong> {result['confidence']:.1f}% {confidence_level}</p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="--width: {result['confidence']}%; background: {color};"></div>
                                </div>
                                <p><strong>Similarity Score:</strong> {result['score']:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Comparison Mode
        elif st.session_state.tab == "Comparison Mode":
            st.markdown("### 📸 Upload Images to Compare")
            
            uploaded_files = st.file_uploader(
                "Drag & drop images to compare",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                accept_multiple_files=True,
                help="Supported formats: JPG, PNG, BMP, TIFF, WebP"
            )
            
            if uploaded_files and len(uploaded_files) >= 2:
                # Display images
                st.markdown("### Images to Compare")
                cols = st.columns(min(4, len(uploaded_files)))
                for i, file in enumerate(uploaded_files):
                    with cols[i % 4]:
                        st.image(file, caption=file.name, use_column_width=True)
                
                # Process button
                if st.button("🔍 Compare Images", type="primary"):
                    if not keywords:
                        st.error("Please enter at least one keyword or caption")
                    else:
                        start_time = time.time()
                        
                        with st.spinner("🤖 Analyzing images..."):
                            # Process each image
                            comparison_results = {}
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, file in enumerate(uploaded_files):
                                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
                                
                                # Save uploaded file temporarily
                                temp_path = None
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                                        f.write(file.getbuffer())
                                        temp_path = f.name
                                    
                                    # Process the image
                                    results = process_image(
                                        temp_path, 
                                        keywords, 
                                        top_n, 
                                        num_candidates, 
                                        st.session_state.caption_method
                                    )
                                    
                                    if results:
                                        comparison_results[file.name] = results
                                    
                                except Exception as e:
                                    st.error(f"Error processing {file.name}: {str(e)}")
                                finally:
                                    # Clean up temporary file
                                    if temp_path and os.path.exists(temp_path):
                                        os.unlink(temp_path)
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(uploaded_files))
                            
                            processing_time = time.time() - start_time
                            st.session_state.comparison_results = comparison_results
                            st.session_state.processing_time = processing_time
                            
                            status_text.text(f"✅ Processed {len(uploaded_files)} images in {processing_time:.2f}s")
            
            # Display comparison results
            if st.session_state.comparison_results:
                st.markdown("### 🎯 Comparison Results")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Images Compared", len(st.session_state.comparison_results))
                with col2:
                    st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
                with col3:
                    avg_conf = np.mean([
                        np.mean([r["confidence"] for r in results])
                        for results in st.session_state.comparison_results.values()
                    ])
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
                # Results visualization
                st.markdown("### 📈 Comparison Visualization")
                
                # Create comparison chart
                fig = create_comparison_chart(st.session_state.comparison_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.markdown("### 📝 Detailed Results")
                
                for img_name, results in st.session_state.comparison_results.items():
                    with st.expander(f"🖼️ {img_name}"):
                        for i, result in enumerate(results, 1):
                            # Confidence level
                            if result["confidence"] >= 80:
                                confidence_level = "🔥 EXCELLENT"
                                color = "#48BB78"
                            elif result["confidence"] >= 60:
                                confidence_level = "✅ GOOD"
                                color = "#4299E1"
                            elif result["confidence"] >= 40:
                                confidence_level = "⚠️ FAIR"
                                color = "#ED8936"
                            else:
                                confidence_level = "❌ POOR"
                                color = "#F56565"
                            
                            # Result card
                            st.markdown(f"""
                            <div class="result-card">
                                <h3>{i}. {result['caption']}</h3>
                                <p><strong>Confidence:</strong> {result['confidence']:.1f}% {confidence_level}</p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="--width: {result['confidence']}%; background: {color};"></div>
                                </div>
                                <p><strong>Similarity Score:</strong> {result['score']:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by CLIP & GPT-2 • Built with Streamlit • © 2023 AI Caption Recommender</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.dark_mode:
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()