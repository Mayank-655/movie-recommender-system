"""
Hybrid Recommendation Model

Combines collaborative filtering and content-based filtering.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.collaborative import CollaborativeFiltering
from models.content_based import ContentBasedFiltering


class HybridRecommender:
    """Hybrid recommendation system combining CF and CB."""
    
    def __init__(self, cf_weight=0.7, cb_weight=0.3):
        """
        Initialize hybrid recommender.
        
        Parameters:
        -----------
        cf_weight : float
            Weight for collaborative filtering (0-1)
        cb_weight : float
            Weight for content-based filtering (0-1)
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = None
        self.cb_model = None
        self.ratings_df = None
        self.movies_df = None
        
        # Normalize weights
        total = cf_weight + cb_weight
        self.cf_weight = cf_weight / total
        self.cb_weight = cb_weight / total
    
    def train(self, ratings_df, movies_df, cf_algorithm='svd'):
        """
        Train hybrid model.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe
        movies_df : pd.DataFrame
            Movies dataframe
        cf_algorithm : str
            Collaborative filtering algorithm
        """
        print(f"\n{'='*60}")
        print("Training Hybrid Recommender")
        print(f"{'='*60}")
        print(f"CF Weight: {self.cf_weight:.2f}, CB Weight: {self.cb_weight:.2f}")
        
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Train collaborative filtering
        print("\n1. Training Collaborative Filtering...")
        self.cf_model = CollaborativeFiltering(algorithm=cf_algorithm)
        self.cf_model.train(ratings_df, algorithm=cf_algorithm)
        
        # Train content-based
        print("\n2. Training Content-Based Filtering...")
        from preprocessing import MoviePreprocessor
        preprocessor = MoviePreprocessor()
        content_features = preprocessor.prepare_content_features(movies_df)
        similarity_matrix = preprocessor.compute_similarity_matrix(content_features)
        
        self.cb_model = ContentBasedFiltering()
        self.cb_model.train(movies_df, similarity_matrix)
        
        print("\n✅ Hybrid model training complete!")
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Get hybrid recommendations for a user.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations
        
        Returns:
        --------
        list
            List of recommended movies
        """
        if self.cf_model is None or self.cb_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Get CF recommendations
        cf_recs = self.cf_model.recommend_for_user(
            user_id, 
            n_recommendations=n_recommendations * 2,  # Get more for filtering
            movies_df=self.movies_df
        )
        
        # Get user's rated movies
        user_ratings = self.ratings_df[
            self.ratings_df['user_id'] == user_id
        ][['movie_id', 'rating']].set_index('movie_id')['rating'].to_dict()
        
        # Get CB recommendations
        cb_recs = self.cb_model.recommend_for_user(
            user_ratings,
            n_recommendations=n_recommendations * 2
        )
        
        # Combine scores
        movie_scores = {}
        
        # Add CF scores
        for rec in cf_recs:
            movie_id = rec['movie_id']
            score = rec['predicted_rating'] / 5.0  # Normalize to 0-1
            movie_scores[movie_id] = score * self.cf_weight
        
        # Add CB scores
        for rec in cb_recs:
            movie_id = rec['movie_id']
            score = rec['score']
            if movie_id in movie_scores:
                movie_scores[movie_id] += score * self.cb_weight
            else:
                movie_scores[movie_id] = score * self.cb_weight
        
        # Sort and get top N
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        user_movie_ids = set(user_ratings.keys())
        
        for movie_id, score in sorted_movies:
            if movie_id not in user_movie_ids:  # Exclude already rated
                movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
                recommendations.append({
                    'movie_id': int(movie_id),
                    'title': movie_info['title'],
                    'hybrid_score': round(score, 4),
                    'genres': movie_info.get('genres', 'N/A')
                })
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations
    
    def save_model(self, filepath='data/models/hybrid_model.pkl'):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'cf_model': self.cf_model.model,
            'cb_model': {
                'similarity_matrix': self.cb_model.similarity_matrix,
                'movies_df': self.cb_model.movies_df
            },
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'ratings_df': self.ratings_df,
            'movies_df': self.movies_df
        }, filepath)
        print(f"✅ Model saved to {filepath}")


def main():
    """Train hybrid model."""
    from data_loader import MovieLensLoader
    
    # Load data
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_all(download=False)
    
    # Train hybrid model
    hybrid = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
    hybrid.train(ratings, movies)
    hybrid.save_model()
    
    # Example recommendation
    print("\n" + "="*60)
    print("Hybrid Recommendations for User 1")
    print("="*60)
    recommendations = hybrid.recommend_for_user(1, n_recommendations=10)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Score: {rec['hybrid_score']:.4f})")


if __name__ == "__main__":
    main()

