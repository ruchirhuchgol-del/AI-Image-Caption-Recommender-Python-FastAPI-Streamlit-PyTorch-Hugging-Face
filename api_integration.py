"""
FastAPI Service for Image Caption Recommendation
Provides RESTful endpoints for batch caption recommendations with enterprise-grade features.
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import contextlib

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic_settings import BaseSettings
import structlog

# Import our recommender system
from caption_recommendation import ImageCaptionRecommendationSystem


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    model_config = ConfigDict(env_file=".env")
    
    # API Configuration
    title: str = "AI Caption Recommendation API"
    description: str = "Enterprise-grade API for image caption recommendations using CLIP embeddings"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    temp_dir: str = "temp_uploads"
    cleanup_interval: int = 3600  # 1 hour in seconds
    
    # Recommendation Configuration
    max_images_per_request: int = 10
    max_keywords_per_request: int = 20
    default_top_n: int = 5
    max_top_n: int = 20
    default_num_candidates: int = 10
    max_num_candidates: int = 50
    
    # Security
    trusted_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"


# Configure structured logging
def setup_logging(settings: Settings) -> None:
    """Configure structured logging with appropriate formatters."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if settings.log_format == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


# Pydantic models with enhanced validation
class RecommendationRequest(BaseModel):
    """Request model for caption recommendations with comprehensive validation."""
    
    image_paths: List[str] = Field(
        ..., 
        min_items=1,
        max_items=10,  # Will be configurable via settings
        example=["image1.jpg", "image2.png"],
        description="List of image file paths"
    )
    keywords: List[str] = Field(
        ..., 
        min_items=1,
        max_items=20,  # Will be configurable via settings
        example=["nature", "landscape", "beautiful"],
        description="List of keywords to include in the caption"
    )
    top_n: int = Field(
        default=5, 
        gt=0, 
        le=20,
        description="Number of top recommendations to return"
    )
    num_candidates: int = Field(
        default=10,
        gt=0,
        le=50,
        description="Number of candidate captions to generate"
    )
    
    @validator('image_paths')
    def validate_image_paths(cls, v):
        """Validate image paths are non-empty and have valid extensions."""
        if not v:
            raise ValueError("At least one image path must be provided")
        
        for path in v:
            if not path.strip():
                raise ValueError("Image path cannot be empty")
                
        return [path.strip() for path in v]
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Validate keywords are non-empty strings."""
        if not v:
            raise ValueError("At least one keyword must be provided")
            
        cleaned_keywords = []
        for keyword in v:
            if not isinstance(keyword, str) or not keyword.strip():
                raise ValueError("Keywords must be non-empty strings")
            cleaned_keywords.append(keyword.strip().lower())
            
        return cleaned_keywords


class RecommendationItem(BaseModel):
    """Individual recommendation item."""
    caption: str = Field(..., description="Generated caption text")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence")


class RecommendationResponse(BaseModel):
    """Response model for caption recommendations."""
    success: bool = Field(True, description="Request success status")
    results: Dict[str, List[RecommendationItem]] = Field(
        ..., 
        description="Mapping of image paths to recommendations"
    )
    processing_time: float = Field(..., gt=0, description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp (ISO format)")
    metadata: Optional[Dict[str, Union[str, int, float]]] = Field(
        None, description="Additional processing metadata"
    )


class UploadResponse(BaseModel):
    """Response model for file upload."""
    success: bool
    file_paths: List[str]
    message: str
    file_info: Optional[List[Dict[str, Union[str, int]]]] = None


class HealthResponse(BaseModel):
    """Health check response with detailed system information."""
    status: str
    model_loaded: bool
    caption_model_loaded: bool
    device: str
    memory_usage: Optional[Dict[str, float]] = None
    uptime_seconds: Optional[float] = None
    timestamp: str


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict] = None
    timestamp: str


# Global application state
class ApplicationState:
    """Centralized application state management."""
    
    def __init__(self):
        self.startup_time = datetime.now(timezone.utc)
        self.recommender: Optional[ImageCaptionRecommendationSystem] = None
        self.settings: Optional[Settings] = None
        
    @property
    def uptime_seconds(self) -> float:
        """Calculate application uptime in seconds."""
        return (datetime.now(timezone.utc) - self.startup_time).total_seconds()


# Dependency injection
def get_settings() -> Settings:
    """Dependency to provide application settings."""
    return app_state.settings


def get_recommender() -> ImageCaptionRecommendationSystem:
    """Dependency to provide recommender system."""
    if app_state.recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender system not initialized"
        )
    return app_state.recommender


# Utility functions
class FileManager:
    """Centralized file management with enhanced validation and cleanup."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.temp_dir = Path(settings.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.logger = structlog.get_logger(__name__)
    
    async def validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file against security and size constraints."""
        # Check content type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type: {file.content_type}. Only images are allowed."
            )
        
        # Check file extension
        if file.filename:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.settings.allowed_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file extension: {file_ext}. Allowed: {self.settings.allowed_extensions}"
                )
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # Reset file pointer
        
        if file_size > self.settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {file_size} bytes. Maximum allowed: {self.settings.max_file_size} bytes"
            )
    
    async def save_file(self, file: UploadFile) -> tuple[str, Dict[str, Union[str, int]]]:
        """Save uploaded file and return path with metadata."""
        await self.validate_file(file)
        
        # Generate unique filename
        file_ext = Path(file.filename).suffix if file.filename else ".tmp"
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = self.temp_dir / unique_filename
        
        try:
            content = await file.read()
            async with asyncio.Lock():
                with file_path.open("wb") as buffer:
                    buffer.write(content)
            
            file_info = {
                "original_name": file.filename or "unknown",
                "size_bytes": len(content),
                "content_type": file.content_type or "unknown"
            }
            
            self.logger.info("File saved successfully", 
                           original_name=file.filename,
                           temp_path=str(file_path),
                           size_bytes=len(content))
            
            return str(file_path), file_info
            
        except Exception as e:
            self.logger.error("Failed to save file", 
                            filename=file.filename, 
                            error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {file.filename}"
            ) from e
    
    async def cleanup_files(self, file_paths: List[str]) -> None:
        """Asynchronously clean up temporary files."""
        for path_str in file_paths:
            try:
                path = Path(path_str)
                if path.exists():
                    path.unlink()
                    self.logger.info("Temporary file deleted", path=path_str)
            except Exception as e:
                self.logger.error("Failed to delete temporary file", 
                                path=path_str, 
                                error=str(e))
    
    async def cleanup_old_files(self) -> None:
        """Clean up files older than cleanup_interval."""
        current_time = datetime.now()
        cleanup_count = 0
        
        for file_path in self.temp_dir.glob("*"):
            try:
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > self.settings.cleanup_interval:
                        file_path.unlink()
                        cleanup_count += 1
            except Exception as e:
                self.logger.error("Error during scheduled cleanup", 
                                file=str(file_path), 
                                error=str(e))
        
        if cleanup_count > 0:
            self.logger.info("Scheduled cleanup completed", files_deleted=cleanup_count)


# Initialize application state and components
app_state = ApplicationState()


def create_app() -> FastAPI:
    """Application factory with proper configuration."""
    settings = Settings()
    app_state.settings = settings
    
    # Setup logging
    setup_logging(settings)
    logger = structlog.get_logger(__name__)
    
    # Initialize FastAPI app
    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
    )
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=settings.trusted_hosts
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize file manager
    file_manager = FileManager(settings)
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        """Enhanced HTTP exception handler with structured logging."""
        logger.error("HTTP exception occurred",
                    status_code=exc.status_code,
                    detail=exc.detail,
                    path=request.url.path)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                error_code=f"HTTP_{exc.status_code}",
                timestamp=datetime.now(timezone.utc).isoformat()
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """Handle unexpected exceptions gracefully."""
        logger.error("Unexpected exception occurred",
                    error=str(exc),
                    path=request.url.path,
                    exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                error_code="INTERNAL_SERVER_ERROR",
                timestamp=datetime.now(timezone.utc).isoformat()
            ).dict()
        )
    
    # API Routes
    @app.get("/health", 
             response_model=HealthResponse, 
             tags=["System"],
             summary="Health check endpoint")
    async def health_check(
        recommender: ImageCaptionRecommendationSystem = Depends(get_recommender)
    ) -> HealthResponse:
        """Comprehensive health check with system metrics."""
        try:
            memory_info = {}
            if hasattr(recommender, 'get_memory_usage'):
                memory_info = recommender.get_memory_usage()
            
            return HealthResponse(
                status="healthy",
                model_loaded=recommender.model is not None,
                caption_model_loaded=getattr(recommender, 'caption_model', None) is not None,
                device=str(recommender.model.device) if recommender.model else "unknown",
                memory_usage=memory_info,
                uptime_seconds=app_state.uptime_seconds,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System health check failed"
            )
    
    @app.post("/upload", 
              response_model=UploadResponse, 
              tags=["File Management"],
              summary="Upload images for processing")
    async def upload_images(
        files: List[UploadFile] = File(...),
        settings: Settings = Depends(get_settings)
    ) -> UploadResponse:
        """Upload multiple images with enhanced validation and metadata."""
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )
        
        if len(files) > settings.max_images_per_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files. Maximum allowed: {settings.max_images_per_request}"
            )
        
        file_paths = []
        file_info_list = []
        
        try:
            for file in files:
                file_path, file_info = await file_manager.save_file(file)
                file_paths.append(file_path)
                file_info_list.append(file_info)
            
            logger.info("Files uploaded successfully", count=len(file_paths))
            
            return UploadResponse(
                success=True,
                file_paths=file_paths,
                message=f"Successfully uploaded {len(file_paths)} images",
                file_info=file_info_list
            )
            
        except HTTPException:
            # Clean up any files that were already saved
            await file_manager.cleanup_files(file_paths)
            raise
        except Exception as e:
            await file_manager.cleanup_files(file_paths)
            logger.error("Upload failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload files"
            ) from e
    
    @app.post("/recommend", 
              response_model=RecommendationResponse, 
              tags=["Recommendations"],
              summary="Get caption recommendations for images")
    async def get_recommendations(
        request: RecommendationRequest,
        background_tasks: BackgroundTasks,
        recommender: ImageCaptionRecommendationSystem = Depends(get_recommender),
        settings: Settings = Depends(get_settings)
    ) -> RecommendationResponse:
        """Get caption recommendations with enhanced error handling and metadata."""
        start_time = datetime.now(timezone.utc)
        
        # Validate request limits
        if len(request.image_paths) > settings.max_images_per_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many images. Maximum: {settings.max_images_per_request}"
            )
        
        if len(request.keywords) > settings.max_keywords_per_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many keywords. Maximum: {settings.max_keywords_per_request}"
            )
        
        # Validate file existence
        missing_files = []
        for path in request.image_paths:
            if not Path(path).exists():
                missing_files.append(path)
        
        if missing_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Images not found: {missing_files}"
            )
        
        try:
            results = {}
            processing_metadata = {
                "total_images": len(request.image_paths),
                "total_keywords": len(request.keywords),
                "successful_predictions": 0,
                "failed_predictions": 0
            }
            
            # Process each image
            for image_path in request.image_paths:
                try:
                    recommendations = recommender.recommend_captions(
                        image_path=image_path,
                        keywords=request.keywords,
                        top_n=request.top_n,
                        num_candidates=request.num_candidates
                    )
                    
                    results[image_path] = [
                        RecommendationItem(
                            caption=caption,
                            score=float(score),
                            confidence=None  # Could be enhanced with model confidence
                        )
                        for caption, score in recommendations
                    ]
                    processing_metadata["successful_predictions"] += 1
                    
                except Exception as e:
                    logger.error("Failed to process image", 
                               image_path=image_path, 
                               error=str(e))
                    results[image_path] = []
                    processing_metadata["failed_predictions"] += 1
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Schedule cleanup
            background_tasks.add_task(file_manager.cleanup_files, request.image_paths)
            
            logger.info("Recommendations generated successfully",
                       processing_time=processing_time,
                       **processing_metadata)
            
            return RecommendationResponse(
                success=True,
                results=results,
                processing_time=processing_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=processing_metadata
            )
            
        except Exception as e:
            logger.error("Recommendation generation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate recommendations"
            ) from e
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("Starting Caption Recommendation API")
        
        try:
            # Initialize recommender system
            app_state.recommender = ImageCaptionRecommendationSystem()
            logger.info("Recommender system initialized successfully")
            
            # Start background cleanup task
            asyncio.create_task(periodic_cleanup(file_manager, settings))
            
        except Exception as e:
            logger.error("Startup failed", error=str(e))
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on application shutdown."""
        logger.info("Shutting down Caption Recommendation API")
        
        try:
            # Final cleanup of temp files
            await file_manager.cleanup_files(
                [str(p) for p in Path(settings.temp_dir).glob("*")]
            )
            logger.info("Shutdown completed successfully")
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))
    
    return app


async def periodic_cleanup(file_manager: FileManager, settings: Settings):
    """Background task for periodic file cleanup."""
    logger = structlog.get_logger(__name__)
    
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval)
            await file_manager.cleanup_old_files()
        except Exception as e:
            logger.error("Periodic cleanup error", error=str(e))


# Create application instance
app = create_app()


if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )