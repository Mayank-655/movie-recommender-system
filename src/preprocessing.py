"""
Data Preprocessing for Movie Recommender

Prepares data for recommendation algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os


class MoviePreprocessor:
    """Preprocess movie data for recommendations."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.tfidf_vectorizer = None
        self.movie_features = None
    
    def prepare_ratings_matrix(self, ratings):
        """
        Create user-item rating matrix.
        
        Parameters:
        -----------
        ratings : pd.DataFrame
            Ratings dataframe
        
        Returns:
        --------
        pd.DataFrame
            User-item matrix (users x movies)
        """
        print("Creating user-item rating matrix...")
        
        # Create pivot table
        rating_matrix = ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        print(f"Matrix shape: {rating_matrix.shape}")
        print(f"Users: {rating_matrix.shape[0]}, Movies: {rating_matrix.shape[1]}")
        
        return rating_matrix
    
    def prepare_content_features(self, movies):
        """
        Prepare content-based features from movie metadata.
        
        Parameters:
        -----------
        movies : pd.DataFrame
            Movies dataframe with genres
        
        Returns:
        --------
        np.array
            Feature matrix for content-based filtering
        """
        print("Preparing content-based features...")
        
        # Use TF-IDF on genres
        if 'genres' in movies.columns:
            self.tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
            self.movie_features = self.tfidf_vectorizer.fit_transform(movies['genres'].fillna(''))
            
            print(f"Feature matrix shape: {self.movie_features.shape}")
            return self.movie_features
        else:
            raise ValueError("Movies dataframe must have 'genres' column")
    
    def compute_similarity_matrix(self, features):
        """
        Compute cosine similarity matrix.
        
        Parameters:
        -----------
        features : np.array or sparse matrix
            Feature matrix
        
        Returns:
        --------
        np.array
            Similarity matrix
        """
        print("Computing similarity matrix...")
        similarity = cosine_similarity(features)
        print(f"Similarity matrix shape: {similarity.shape}")
        return similarity
    
    def save_preprocessor(self, filepath='data/models/preprocessor.pkl'):
        """Save preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'movie_features': self.movie_features
        }, filepath)
        print(f"✅ Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='data/models/preprocessor.pkl'):
        """Load preprocessor from disk."""
        preprocessor = joblib.load(filepath)
        self.tfidf_vectorizer = preprocessor['tfidf_vectorizer']
        self.movie_features = preprocessor['movie_features']
        print(f"✅ Preprocessor loaded from {filepath}")


def main():
    """Example usage."""
    from data_loader import MovieLensLoader
    
    # Load data
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_all(download=False)
    
    # Preprocess
    preprocessor = MoviePreprocessor()
    
    # Create rating matrix
    rating_matrix = preprocessor.prepare_ratings_matrix(ratings)
    
    # Prepare content features
    content_features = preprocessor.prepare_content_features(movies)
    
    # Compute similarity
    similarity_matrix = preprocessor.compute_similarity_matrix(content_features)
    
    # Save
    preprocessor.save_preprocessor()
    rating_matrix.to_csv('data/processed/rating_matrix.csv')
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()

