"""
Data Visualization Module
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer:
    """Visualize movie and rating data."""
    
    def __init__(self):
        """Initialize visualizer."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_rating_distribution(self, ratings_df, save_path=None):
        """Plot rating distribution."""
        plt.figure(figsize=(10, 6))
        ratings_df['rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('Rating Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_movies(self, ratings_df, movies_df, n=20, save_path=None):
        """Plot top rated movies."""
        # Calculate average ratings
        movie_ratings = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
        movie_ratings = movie_ratings[movie_ratings['count'] >= 10]  # At least 10 ratings
        movie_ratings = movie_ratings.sort_values('mean', ascending=False).head(n)
        
        # Merge with movie titles
        top_movies = movie_ratings.merge(movies_df, on='movie_id')
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_movies)), top_movies['mean'], color='coral')
        plt.yticks(range(len(top_movies)), top_movies['title'], fontsize=10)
        plt.xlabel('Average Rating', fontsize=12)
        plt.title(f'Top {n} Rated Movies', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_genre_distribution(self, movies_df, save_path=None):
        """Plot genre distribution."""
        # Extract all genres
        all_genres = []
        for genres in movies_df['genres'].dropna():
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        
        plt.figure(figsize=(12, 6))
        genre_counts.plot(kind='bar', color='lightgreen')
        plt.title('Movie Genre Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Number of Movies', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_user_activity(self, ratings_df, save_path=None):
        """Plot user rating activity."""
        user_ratings = ratings_df.groupby('user_id').size()
        
        plt.figure(figsize=(10, 6))
        plt.hist(user_ratings, bins=50, edgecolor='black', color='plum')
        plt.title('User Rating Activity', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Ratings per User', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

