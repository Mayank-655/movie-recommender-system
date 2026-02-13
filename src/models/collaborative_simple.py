"""
Simple Collaborative Filtering (without Surprise library)

Alternative implementation using only scikit-learn and NumPy.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import MovieLensLoader


class SimpleCollaborativeFiltering:
    """Simple collaborative filtering using matrix factorization."""
    
    def __init__(self, n_components=50):
        """
        Initialize collaborative filtering model.
        
        Parameters:
        -----------
        n_components : int
            Number of latent factors
        """
        self.n_components = n_components
        self.model = None
        self.rating_matrix = None
        self.user_ids = None
        self.movie_ids = None
        
    def prepare_rating_matrix(self, ratings):
        """Create user-item rating matrix."""
        print("Creating rating matrix...")
        
        self.rating_matrix = ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        self.user_ids = self.rating_matrix.index.values
        self.movie_ids = self.rating_matrix.columns.values
        
        print(f"Matrix shape: {self.rating_matrix.shape}")
        return self.rating_matrix
    
    def train(self, ratings):
        """
        Train collaborative filtering model using NMF.
        
        Parameters:
        -----------
        ratings : pd.DataFrame
            Ratings dataframe
        """
        print(f"\n{'='*60}")
        print("Training Simple Collaborative Filtering (NMF)")
        print(f"{'='*60}")
        
        # Prepare matrix
        self.prepare_rating_matrix(ratings)
        
        # Normalize ratings to 0-1 scale
        matrix_normalized = self.rating_matrix / 5.0
        
        # Train NMF model
        print("Training NMF model...")
        self.model = NMF(n_components=self.n_components, random_state=42, max_iter=200)
        W = self.model.fit_transform(matrix_normalized)
        H = self.model.components_
        
        # Reconstruct ratings
        reconstructed = np.dot(W, H) * 5.0
        
        # Calculate RMSE on training data
        mask = self.rating_matrix.values > 0
        rmse = np.sqrt(np.mean((reconstructed[mask] - self.rating_matrix.values[mask]) ** 2))
        
        print(f"✅ Training complete!")
        print(f"RMSE: {rmse:.4f}")
        
        return self.model
    
    def recommend_for_user(self, user_id, n_recommendations=10, movies_df=None):
        """
        Get recommendations for a user.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations
        movies_df : pd.DataFrame, optional
            Movies dataframe
        
        Returns:
        --------
        list
            List of recommended movies
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if user_id not in self.user_ids:
            raise ValueError(f"User ID {user_id} not found in training data")
        
        # Get user index
        user_idx = np.where(self.user_ids == user_id)[0][0]
        
        # Get user's predicted ratings
        matrix_normalized = self.rating_matrix / 5.0
        W = self.model.transform(matrix_normalized)
        H = self.model.components_
        user_ratings = np.dot(W[user_idx], H) * 5.0
        
        # Get user's already rated movies
        user_rated = self.rating_matrix.iloc[user_idx]
        rated_movie_ids = user_rated[user_rated > 0].index.values
        
        # Get top recommendations (excluding already rated)
        movie_scores = []
        for i, movie_id in enumerate(self.movie_ids):
            if movie_id not in rated_movie_ids:
                movie_scores.append((movie_id, user_ratings[i]))
        
        # Sort by predicted rating
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        recommendations = movie_scores[:n_recommendations]
        
        # Add movie titles if available
        if movies_df is not None:
            result = []
            for movie_id, rating in recommendations:
                movie_info = movies_df[movies_df['movie_id'] == movie_id]
                if not movie_info.empty:
                    result.append({
                        'movie_id': int(movie_id),
                        'title': movie_info.iloc[0]['title'],
                        'predicted_rating': round(float(rating), 2)
                    })
            return result
        
        return recommendations
    
    def save_model(self, filepath='data/models/collaborative_simple.pkl'):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'rating_matrix': self.rating_matrix,
            'user_ids': self.user_ids,
            'movie_ids': self.movie_ids,
            'n_components': self.n_components
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='data/models/collaborative_simple.pkl'):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.rating_matrix = model_data['rating_matrix']
        self.user_ids = model_data['user_ids']
        self.movie_ids = model_data['movie_ids']
        self.n_components = model_data['n_components']
        print(f"✅ Model loaded from {filepath}")


def main():
    """Train and test simple collaborative filtering."""
    # Load data
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_all(download=False)
    
    # Train model
    cf = SimpleCollaborativeFiltering(n_components=50)
    cf.train(ratings)
    cf.save_model()
    
    # Example recommendation
    print("\n" + "="*60)
    print("Example Recommendation for User 1")
    print("="*60)
    recommendations = cf.recommend_for_user(1, n_recommendations=10, movies_df=movies)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Predicted Rating: {rec['predicted_rating']})")


if __name__ == "__main__":
    main()

