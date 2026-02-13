"""
MovieLens Dataset Loader

Downloads and loads the MovieLens 100K dataset for movie recommendations.
"""

import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
from pathlib import Path


class MovieLensLoader:
    """Load MovieLens 100K dataset."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.base_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, force_download=False):
        """
        Download MovieLens 100K dataset.
        
        Parameters:
        -----------
        force_download : bool
            Force re-download even if files exist
        
        Returns:
        --------
        str
            Path to downloaded dataset
        """
        zip_path = self.data_dir / "ml-100k.zip"
        extract_path = self.data_dir / "ml-100k"
        
        # Check if already downloaded
        if extract_path.exists() and not force_download:
            print("✅ Dataset already downloaded")
            return str(extract_path)
        
        print("📥 Downloading MovieLens 100K dataset...")
        print(f"Source: {self.base_url}")
        
        try:
            # Download zip file
            urllib.request.urlretrieve(self.base_url, zip_path)
            print(f"✅ Downloaded to {zip_path}")
            
            # Extract zip file
            print("📦 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            print(f"✅ Extracted to {extract_path}")
            return str(extract_path)
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            raise
    
    def load_ratings(self, dataset_path=None):
        """
        Load ratings data.
        
        Parameters:
        -----------
        dataset_path : str, optional
            Path to dataset directory
        
        Returns:
        --------
        pd.DataFrame
            Ratings dataframe with columns: user_id, movie_id, rating, timestamp
        """
        if dataset_path is None:
            dataset_path = self.data_dir / "ml-100k"
        
        ratings_file = Path(dataset_path) / "u.data"
        
        if not ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
        
        print(f"📊 Loading ratings from {ratings_file}")
        
        # Load ratings (tab-separated, no header)
        ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        print(f"✅ Loaded {len(ratings):,} ratings")
        return ratings
    
    def load_movies(self, dataset_path=None):
        """
        Load movies data.
        
        Parameters:
        -----------
        dataset_path : str, optional
            Path to dataset directory
        
        Returns:
        --------
        pd.DataFrame
            Movies dataframe with columns: movie_id, title, release_date, genres
        """
        if dataset_path is None:
            dataset_path = self.data_dir / "ml-100k"
        
        movies_file = Path(dataset_path) / "u.item"
        genres_file = Path(dataset_path) / "u.genre"
        
        if not movies_file.exists():
            raise FileNotFoundError(f"Movies file not found: {movies_file}")
        
        print(f"🎬 Loading movies from {movies_file}")
        
        # Load genres
        genres = []
        if genres_file.exists():
            with open(genres_file, 'r', encoding='latin-1') as f:
                for line in f:
                    if line.strip():
                        genres.append(line.strip().split('|')[0])
        
        # Load movies (pipe-separated, no header)
        movies = pd.read_csv(
            movies_file,
            sep='|',
            header=None,
            encoding='latin-1',
            names=[
                'movie_id', 'title', 'release_date', 'video_release_date',
                'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )
        
        # Extract genres
        genre_cols = [col for col in movies.columns if col not in 
                     ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']]
        
        movies['genres'] = movies[genre_cols].apply(
            lambda row: '|'.join([col for col in genre_cols if row[col] == 1]),
            axis=1
        )
        
        # Drop individual genre columns
        movies = movies[['movie_id', 'title', 'release_date', 'genres']]
        
        print(f"✅ Loaded {len(movies):,} movies")
        return movies
    
    def load_users(self, dataset_path=None):
        """
        Load users data.
        
        Parameters:
        -----------
        dataset_path : str, optional
            Path to dataset directory
        
        Returns:
        --------
        pd.DataFrame
            Users dataframe
        """
        if dataset_path is None:
            dataset_path = self.data_dir / "ml-100k"
        
        users_file = Path(dataset_path) / "u.user"
        
        if not users_file.exists():
            print("⚠️  Users file not found, skipping...")
            return None
        
        print(f"👥 Loading users from {users_file}")
        
        users = pd.read_csv(
            users_file,
            sep='|',
            header=None,
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        print(f"✅ Loaded {len(users):,} users")
        return users
    
    def load_all(self, download=True):
        """
        Load all MovieLens data.
        
        Parameters:
        -----------
        download : bool
            Download dataset if not present
        
        Returns:
        --------
        tuple
            (ratings, movies, users) dataframes
        """
        if download:
            dataset_path = self.download_dataset()
        else:
            dataset_path = self.data_dir / "ml-100k"
        
        ratings = self.load_ratings(dataset_path)
        movies = self.load_movies(dataset_path)
        users = self.load_users(dataset_path)
        
        return ratings, movies, users


def main():
    """Download and load MovieLens dataset."""
    loader = MovieLensLoader()
    
    print("="*60)
    print("MovieLens 100K Dataset Loader")
    print("="*60)
    
    ratings, movies, users = loader.load_all(download=True)
    
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Ratings: {len(ratings):,}")
    print(f"Movies: {len(movies):,}")
    users_count = len(users) if users is not None else 'N/A'
    print(f"Users: {users_count:,}" if isinstance(users_count, int) else f"Users: {users_count}")
    print(f"\nRating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    ratings.to_csv("data/processed/ratings.csv", index=False)
    movies.to_csv("data/processed/movies.csv", index=False)
    if users is not None:
        users.to_csv("data/processed/users.csv", index=False)
    
    print("✅ Data saved to data/processed/")


if __name__ == "__main__":
    main()

