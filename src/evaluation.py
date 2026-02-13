"""
Model Evaluation Module
"""

import pandas as pd
import numpy as np
from surprise import accuracy
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate recommendation models."""
    
    def __init__(self):
        """Initialize evaluator."""
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def evaluate_collaborative(self, predictions):
        """
        Evaluate collaborative filtering model.
        
        Parameters:
        -----------
        predictions : list
            List of prediction objects from Surprise
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }
    
    def precision_at_k(self, recommendations, actual_ratings, k=10, threshold=3.5):
        """
        Calculate Precision@K.
        
        Parameters:
        -----------
        recommendations : list
            List of recommended movie IDs
        actual_ratings : pd.DataFrame
            Actual user ratings
        k : int
            Number of recommendations
        threshold : float
            Rating threshold for relevance
        
        Returns:
        --------
        float
            Precision@K
        """
        # Get top K recommendations
        top_k = recommendations[:k]
        
        # Get relevant items (rated above threshold)
        relevant = set(actual_ratings[actual_ratings['rating'] >= threshold]['movie_id'].values)
        
        # Count relevant recommendations
        recommended_relevant = len(set(top_k) & relevant)
        
        return recommended_relevant / k if k > 0 else 0
    
    def recall_at_k(self, recommendations, actual_ratings, k=10, threshold=3.5):
        """
        Calculate Recall@K.
        
        Parameters:
        -----------
        recommendations : list
            List of recommended movie IDs
        actual_ratings : pd.DataFrame
            Actual user ratings
        k : int
            Number of recommendations
        threshold : float
            Rating threshold for relevance
        
        Returns:
        --------
        float
            Recall@K
        """
        # Get top K recommendations
        top_k = recommendations[:k]
        
        # Get relevant items
        relevant = set(actual_ratings[actual_ratings['rating'] >= threshold]['movie_id'].values)
        
        if len(relevant) == 0:
            return 0
        
        # Count relevant recommendations
        recommended_relevant = len(set(top_k) & relevant)
        
        return recommended_relevant / len(relevant)
    
    def plot_rating_distribution(self, ratings_df, save_path=None):
        """Plot rating distribution."""
        plt.figure(figsize=(10, 6))
        ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_user_activity(self, ratings_df, save_path=None):
        """Plot user rating activity."""
        user_ratings = ratings_df.groupby('user_id').size()
        
        plt.figure(figsize=(10, 6))
        plt.hist(user_ratings, bins=50, edgecolor='black')
        plt.title('User Rating Activity')
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Number of Users')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

