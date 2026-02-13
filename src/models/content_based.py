"""
Content-Based Recommendation Model

Recommends movies based on movie features (genres, etc.).
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import MoviePreprocessor
from data_loader import MovieLensLoader


class ContentBasedFiltering:
    """Content-based recommendation system."""
    
    def __init__(self):
        """Initialize content-based model."""
        self.similarity_matrix = None
        self.movies_df = None
        self.preprocessor = None
    
    def train(self, movies_df, similarity_matrix=None):
        """
        Train content-based model.
        
        Parameters:
        -----------
        movies_df : pd.DataFrame
            Movies dataframe
        similarity_matrix : np.array, optional
            Pre-computed similarity matrix
        """
        print(f"\n{'='*60}")
        print("Training Content-Based Filtering")
        print(f"{'='*60}")
        
        self.movies_df = movies_df.copy()
        
        if similarity_matrix is not None:
            self.similarity_matrix = similarity_matrix
        else:
            # Compute similarity from preprocessor
            preprocessor = MoviePreprocessor()
            features = preprocessor.prepare_content_features(movies_df)
            self.similarity_matrix = preprocessor.compute_similarity_matrix(features)
        
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        print("✅ Training complete!")
        
        return self.similarity_matrix
    
    def recommend_similar_movies(self, movie_id, n_recommendations=10):
        """
        Get similar movies to a given movie.
        
        Parameters:
        -----------
        movie_id : int
            Movie ID
        n_recommendations : int
            Number of recommendations
        
        Returns:
        --------
        list
            List of similar movies
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find movie index
        movie_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index
        
        if len(movie_idx) == 0:
            raise ValueError(f"Movie ID {movie_id} not found")
        
        movie_idx = movie_idx[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N (excluding the movie itself)
        top_movies = similarity_scores[1:n_recommendations+1]
        
        # Get movie details
        recommendations = []
        for idx, score in top_movies:
            movie_info = self.movies_df.iloc[idx]
            recommendations.append({
                'movie_id': int(movie_info['movie_id']),
                'title': movie_info['title'],
                'similarity_score': round(float(score), 4),
                'genres': movie_info.get('genres', 'N/A')
            })
        
        return recommendations
    
    def recommend_for_user(self, user_ratings, n_recommendations=10):
        """
        Get recommendations based on user's rated movies.
        
        Parameters:
        -----------
        user_ratings : dict
            Dictionary of {movie_id: rating}
        n_recommendations : int
            Number of recommendations
        
        Returns:
        --------
        list
            List of recommended movies
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Calculate weighted average similarity
        movie_scores = {}
        
        for movie_id, rating in user_ratings.items():
            # Get similar movies
            similar_movies = self.recommend_similar_movies(movie_id, n_recommendations=50)
            
            for movie in similar_movies:
                movie_id_rec = movie['movie_id']
                similarity = movie['similarity_score']
                
                # Weight by user's rating and similarity
                if movie_id_rec not in movie_scores:
                    movie_scores[movie_id_rec] = 0
                movie_scores[movie_id_rec] += rating * similarity
        
        # Sort by score
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N (excluding already rated movies)
        recommendations = []
        for movie_id, score in sorted_movies:
            if movie_id not in user_ratings:
                movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
                recommendations.append({
                    'movie_id': int(movie_id),
                    'title': movie_info['title'],
                    'score': round(score, 4),
                    'genres': movie_info.get('genres', 'N/A')
                })
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations
    
    def save_model(self, filepath='data/models/content_based_model.pkl'):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'similarity_matrix': self.similarity_matrix,
            'movies_df': self.movies_df
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='data/models/content_based_model.pkl'):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.similarity_matrix = model_data['similarity_matrix']
        self.movies_df = model_data['movies_df']
        print(f"✅ Model loaded from {filepath}")


def main():
    """Train and test content-based model."""
    # Load data
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_all(download=False)
    
    # Preprocess
    preprocessor = MoviePreprocessor()
    content_features = preprocessor.prepare_content_features(movies)
    similarity_matrix = preprocessor.compute_similarity_matrix(content_features)
    
    # Train model
    cb = ContentBasedFiltering()
    cb.train(movies, similarity_matrix)
    cb.save_model()
    
    # Example: Similar movies to movie ID 1
    print("\n" + "="*60)
    print("Similar Movies to Movie ID 1")
    print("="*60)
    movie_title = movies[movies['movie_id'] == 1]['title'].iloc[0]
    print(f"Movie: {movie_title}\n")
    
    recommendations = cb.recommend_similar_movies(1, n_recommendations=10)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Similarity: {rec['similarity_score']:.4f})")


if __name__ == "__main__":
    main()

