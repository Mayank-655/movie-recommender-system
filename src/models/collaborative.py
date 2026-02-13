"""
Collaborative Filtering Recommendation Model

Implements user-based and item-based collaborative filtering.
"""

import pandas as pd
import numpy as np
try:
    from surprise import Dataset, Reader, KNNBasic, SVD, accuracy
    from surprise.model_selection import train_test_split, cross_validate
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("⚠️  Surprise library not available. Using alternative implementation.")
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import MovieLensLoader


class CollaborativeFiltering:
    """Collaborative filtering recommendation system."""
    
    def __init__(self, algorithm='svd'):
        """
        Initialize collaborative filtering model.
        
        Parameters:
        -----------
        algorithm : str
            Algorithm to use ('knn', 'svd')
        """
        self.algorithm = algorithm
        self.model = None
        self.trainset = None
        self.testset = None
        
    def prepare_data(self, ratings):
        """
        Prepare data for Surprise library.
        
        Parameters:
        -----------
        ratings : pd.DataFrame
            Ratings dataframe
        """
        print("Preparing data for collaborative filtering...")
        
        # Surprise requires: user, item, rating
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings[['user_id', 'movie_id', 'rating']],
            reader
        )
        
        # Split data
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)
        print(f"Training set: {self.trainset.n_users} users, {self.trainset.n_items} items")
        print(f"Test set: {len(self.testset)} ratings")
        
        return self.trainset, self.testset
    
    def train(self, ratings, algorithm='svd'):
        if not SURPRISE_AVAILABLE:
            raise ImportError("Surprise library required. Install with: pip install scikit-surprise\nOr use collaborative_simple.py for alternative implementation")
        """
        Train collaborative filtering model.
        
        Parameters:
        -----------
        ratings : pd.DataFrame
            Ratings dataframe
        algorithm : str
            Algorithm to use
        """
        print(f"\n{'='*60}")
        print(f"Training Collaborative Filtering ({algorithm.upper()})")
        print(f"{'='*60}")
        
        # Prepare data
        self.prepare_data(ratings)
        
        # Choose algorithm
        if algorithm == 'knn':
            self.model = KNNBasic(
                k=40,
                min_k=1,
                sim_options={'name': 'cosine', 'user_based': True}
            )
        elif algorithm == 'svd':
            self.model = SVD(n_factors=50, n_epochs=20, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train
        print("Training model...")
        self.model.fit(self.trainset)
        print("✅ Training complete!")
        
        # Evaluate
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        print(f"\nModel Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
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
            Movies dataframe for movie titles
        
        Returns:
        --------
        list
            List of recommended movies
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get all movies
        all_movies = self.trainset.all_items()
        user_inner_id = self.trainset.to_inner_uid(user_id)
        
        # Predict ratings for all movies
        predictions = []
        for movie_id in all_movies:
            movie_raw_id = self.trainset.to_raw_iid(movie_id)
            pred = self.model.predict(user_id, movie_raw_id)
            predictions.append((movie_raw_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        recommendations = predictions[:n_recommendations]
        
        # Add movie titles if available
        if movies_df is not None:
            result = []
            for movie_id, rating in recommendations:
                movie_info = movies_df[movies_df['movie_id'] == movie_id]
                if not movie_info.empty:
                    result.append({
                        'movie_id': int(movie_id),
                        'title': movie_info.iloc[0]['title'],
                        'predicted_rating': round(rating, 2)
                    })
            return result
        
        return recommendations
    
    def save_model(self, filepath='data/models/collaborative_model.pkl'):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='data/models/collaborative_model.pkl'):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")


def main():
    """Train and evaluate collaborative filtering model."""
    # Load data
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_all(download=False)
    
    # Train SVD model
    cf_svd = CollaborativeFiltering(algorithm='svd')
    cf_svd.train(ratings, algorithm='svd')
    cf_svd.save_model('data/models/collaborative_svd.pkl')
    
    # Train KNN model
    cf_knn = CollaborativeFiltering(algorithm='knn')
    cf_knn.train(ratings, algorithm='knn')
    cf_knn.save_model('data/models/collaborative_knn.pkl')
    
    # Example recommendation
    print("\n" + "="*60)
    print("Example Recommendation for User 1")
    print("="*60)
    recommendations = cf_svd.recommend_for_user(1, n_recommendations=10, movies_df=movies)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Predicted Rating: {rec['predicted_rating']})")


if __name__ == "__main__":
    main()

