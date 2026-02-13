"""
FastAPI Application for Movie Recommender System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Use simple collaborative filtering (works without Surprise library)
from src.models.collaborative_simple import SimpleCollaborativeFiltering as CollaborativeFiltering
from src.models.content_based import ContentBasedFiltering
from src.models.hybrid import HybridRecommender
from src.data_loader import MovieLensLoader

app = FastAPI(
    title="Movie Recommender API",
    description="REST API for movie recommendations using collaborative and content-based filtering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
models = {}
movies_df = None
ratings_df = None

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global models, movies_df, ratings_df
    
    try:
        # Load data
        loader = MovieLensLoader()
        ratings_df, movies_df, _ = loader.load_all(download=False)
        
        # Load models if they exist
        if os.path.exists('data/models/collaborative_simple.pkl'):
            cf = CollaborativeFiltering()
            cf.load_model('data/models/collaborative_simple.pkl')
            models['collaborative'] = cf
        
        if os.path.exists('data/models/content_based_model.pkl'):
            cb = ContentBasedFiltering()
            cb.load_model('data/models/content_based_model.pkl')
            models['content_based'] = cb
        
        print(f"✅ Loaded {len(models)} model(s)")
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    movie_id: Optional[int] = None
    n_recommendations: int = 10
    algorithm: str = "hybrid"  # collaborative, content_based, hybrid

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Movie Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/recommend/user/{user_id}": "Get recommendations for user",
            "/recommend/movie/{movie_id}": "Get similar movies",
            "/movies/popular": "Get popular movies",
            "/movies/{movie_id}": "Get movie details"
        }
    }

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "movies": len(movies_df) if movies_df is not None else 0
    }

@app.get("/recommend/user/{user_id}")
async def recommend_for_user(user_id: int, n_recommendations: int = 10, algorithm: str = "collaborative"):
    """Get recommendations for a user."""
    if 'collaborative' not in models:
        raise HTTPException(status_code=503, detail="Collaborative model not loaded")
    
    try:
        recommendations = models['collaborative'].recommend_for_user(
            user_id, 
            n_recommendations=n_recommendations,
            movies_df=movies_df
        )
        return {
            "user_id": user_id,
            "algorithm": algorithm,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/movie/{movie_id}")
async def recommend_similar_movies(movie_id: int, n_recommendations: int = 10):
    """Get similar movies."""
    if 'content_based' not in models:
        raise HTTPException(status_code=503, detail="Content-based model not loaded")
    
    try:
        recommendations = models['content_based'].recommend_similar_movies(
            movie_id,
            n_recommendations=n_recommendations
        )
        return {
            "movie_id": movie_id,
            "algorithm": "content_based",
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies/popular")
async def get_popular_movies(n: int = 10):
    """Get popular movies by rating."""
    if ratings_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    popular = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
    popular = popular[popular['count'] >= 10]  # At least 10 ratings
    popular = popular.sort_values('mean', ascending=False).head(n)
    
    result = []
    for _, row in popular.iterrows():
        movie_info = movies_df[movies_df['movie_id'] == row['movie_id']].iloc[0]
        result.append({
            'movie_id': int(row['movie_id']),
            'title': movie_info['title'],
            'average_rating': round(row['mean'], 2),
            'rating_count': int(row['count'])
        })
    
    return {"popular_movies": result}

@app.get("/movies/{movie_id}")
async def get_movie_details(movie_id: int):
    """Get movie details."""
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    movie = movies_df[movies_df['movie_id'] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_info = movie.iloc[0]
    
    # Get rating stats
    movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]['rating']
    
    return {
        "movie_id": int(movie_id),
        "title": movie_info['title'],
        "release_date": movie_info.get('release_date', 'N/A'),
        "genres": movie_info.get('genres', 'N/A'),
        "average_rating": round(movie_ratings.mean(), 2) if len(movie_ratings) > 0 else None,
        "rating_count": len(movie_ratings)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

