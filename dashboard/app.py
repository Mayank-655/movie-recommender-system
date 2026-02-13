"""
Netflix-Style Movie Recommender Dashboard
"""

import streamlit as st
import pandas as pd
import sys
import os
import requests
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import MovieLensLoader
from src.models.collaborative_simple import SimpleCollaborativeFiltering as CollaborativeFiltering
from src.models.content_based import ContentBasedFiltering

# Page config
st.set_page_config(
    page_title="Movie Recommender - Netflix Style",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Netflix-style design
st.markdown("""
<style>
    /* Netflix-style dark theme */
    .stApp {
        background: linear-gradient(180deg, #141414 0%, #000000 100%);
        color: #ffffff;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom movie card styling */
    .movie-card {
        background: #1f1f1f;
        border-radius: 8px;
        padding: 15px;
        margin: 10px;
        transition: transform 0.3s, box-shadow 0.3s;
        border: 1px solid #333;
        height: 100%;
    }
    
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(229, 9, 20, 0.3);
        border-color: #e50914;
    }
    
    .movie-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        margin-top: 10px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .movie-rating {
        color: #e50914;
        font-size: 14px;
        margin-top: 5px;
    }
    
    .movie-genres {
        color: #b3b3b3;
        font-size: 12px;
        margin-top: 5px;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(180deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.4) 100%);
        padding: 60px 20px;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        color: #e50914;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #ffffff;
        margin-bottom: 30px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ffffff;
        margin: 30px 0 20px 0;
        padding-left: 10px;
        border-left: 4px solid #e50914;
    }
    
    /* Buttons */
    .stButton > button {
        background: #e50914;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 30px;
        font-weight: bold;
        transition: background 0.3s;
    }
    
    .stButton > button:hover {
        background: #f40612;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        background: #1f1f1f;
        color: white;
        border: 1px solid #333;
    }
    
    .stSelectbox > div > div > select {
        background: #1f1f1f;
        color: white;
    }
    
    /* Movie grid */
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 20px;
        padding: 20px 0;
    }
    
    /* Scrollable container */
    .scrollable-container {
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        padding: 20px 0;
        -webkit-overflow-scrolling: touch;
    }
    
    .movie-row {
        display: inline-flex;
        gap: 15px;
    }
    
    /* Rating stars */
    .rating-stars {
        color: #ffd700;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load movie data."""
    loader = MovieLensLoader()
    # Try to load, if fails, download automatically
    try:
        ratings, movies, users = loader.load_all(download=False)
    except FileNotFoundError:
        st.info("📥 Downloading dataset for the first time... This may take a minute.")
        ratings, movies, users = loader.load_all(download=True)
    return ratings, movies, users

@st.cache_resource
def load_models():
    """Load recommendation models."""
    models = {}
    
    if os.path.exists('data/models/collaborative_simple.pkl'):
        cf = CollaborativeFiltering()
        cf.load_model('data/models/collaborative_simple.pkl')
        models['collaborative'] = cf
    
    if os.path.exists('data/models/content_based_model.pkl'):
        cb = ContentBasedFiltering()
        cb.load_model('data/models/content_based_model.pkl')
        models['content_based'] = cb
    
    return models

def create_movie_card(movie, rating=None, similarity=None, index=0):
    """Create a Netflix-style movie card."""
    card_html = f"""
    <div class="movie-card" style="min-width: 180px; max-width: 200px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    height: 250px; 
                    border-radius: 8px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    font-size: 48px;
                    color: white;
                    margin-bottom: 10px;">
            🎬
        </div>
        <div class="movie-title">{movie.get('title', 'Unknown Movie')}</div>
    """
    
    if rating:
        stars = "⭐" * int(rating)
        card_html += f'<div class="movie-rating">{stars} {rating:.1f}/5.0</div>'
    
    if similarity:
        card_html += f'<div class="movie-genres">Similarity: {similarity:.2f}</div>'
    
    if 'genres' in movie and movie['genres']:
        genres = movie['genres'].split('|')[:2]  # Show first 2 genres
        card_html += f'<div class="movie-genres">{", ".join(genres)}</div>'
    
    card_html += "</div>"
    return card_html

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">MOVIE RECOMMENDER</div>
        <div class="hero-subtitle">Discover Your Next Favorite Movie</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        ratings, movies, users = load_data()
        models = load_models()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("📥 Attempting to download dataset automatically...")
        try:
            loader = MovieLensLoader()
            ratings, movies, users = loader.load_all(download=True)
            models = load_models()
            st.success("✅ Dataset downloaded successfully! Please refresh the page.")
            st.rerun()
        except Exception as download_error:
            st.error(f"Download failed: {download_error}")
            st.info("Please wait a moment and refresh the page, or check your internet connection.")
        return
    
    # Sidebar (minimal, Netflix-style)
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        page = st.radio("", [
            "🏠 Home",
            "👤 For You",
            "🔍 Discover",
            "⭐ Popular",
            "🎭 Similar Movies"
        ])
    
    # Home Page
    if page == "🏠 Home":
        st.markdown('<div class="section-header">🎬 Trending Now</div>', unsafe_allow_html=True)
        
        # Popular movies
        popular = ratings.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
        popular = popular[popular['count'] >= 20]
        popular = popular.sort_values('mean', ascending=False).head(20)
        popular_movies = popular.merge(movies, on='movie_id')
        
        # Create scrollable movie row
        movie_cards = ""
        for _, row in popular_movies.iterrows():
            movie_cards += create_movie_card(
                {'title': row['title'], 'genres': row.get('genres', '')},
                rating=row['mean']
            )
        
        st.markdown(f'<div class="scrollable-container"><div class="movie-row">{movie_cards}</div></div>', unsafe_allow_html=True)
        
        # Top rated section
        st.markdown('<div class="section-header">⭐ Top Rated</div>', unsafe_allow_html=True)
        top_rated = popular_movies.head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(top_rated.iterrows()):
            with cols[idx % 5]:
                st.markdown(create_movie_card(
                    {'title': row['title'], 'genres': row.get('genres', '')},
                    rating=row['mean']
                ), unsafe_allow_html=True)
    
    # For You (User Recommendations)
    elif page == "👤 For You":
        st.markdown('<div class="section-header">🎯 Personalized Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, key="user_rec")
            n_recs = st.slider("Number of recommendations", 5, 20, 10, key="n_recs")
            algorithm = st.selectbox("Algorithm", ["collaborative", "content_based"], key="algo")
        
        if st.button("🎬 Get Recommendations", key="get_recs"):
            if algorithm == "collaborative" and 'collaborative' in models:
                with st.spinner("🎭 Finding perfect movies for you..."):
                    recommendations = models['collaborative'].recommend_for_user(
                        user_id, n_recommendations=n_recs, movies_df=movies
                    )
                
                st.success(f"✨ Found {len(recommendations)} personalized recommendations!")
                st.markdown('<div class="section-header">Your Recommendations</div>', unsafe_allow_html=True)
                
                # Display in grid
                cols = st.columns(5)
                for idx, rec in enumerate(recommendations):
                    with cols[idx % 5]:
                        st.markdown(create_movie_card(
                            {'title': rec['title'], 'genres': movies[movies['movie_id'] == rec['movie_id']]['genres'].iloc[0] if not movies[movies['movie_id'] == rec['movie_id']].empty else ''},
                            rating=rec['predicted_rating']
                        ), unsafe_allow_html=True)
            else:
                st.error("⚠️ Model not available. Please train models first.")
    
    # Discover (Movie Search)
    elif page == "🔍 Discover":
        st.markdown('<div class="section-header">🔍 Discover Movies</div>', unsafe_allow_html=True)
        
        search_term = st.text_input("", placeholder="Search for movies...", key="search")
        
        if search_term:
            results = movies[movies['title'].str.contains(search_term, case=False, na=False)]
            
            if len(results) > 0:
                st.success(f"Found {len(results)} movies")
                
                # Display results in grid
                cols = st.columns(5)
                for idx, (_, row) in enumerate(results.iterrows()):
                    with cols[idx % 5]:
                        # Get rating if available
                        movie_ratings = ratings[ratings['movie_id'] == row['movie_id']]['rating']
                        avg_rating = movie_ratings.mean() if len(movie_ratings) > 0 else None
                        
                        st.markdown(create_movie_card(
                            {'title': row['title'], 'genres': row.get('genres', '')},
                            rating=avg_rating
                        ), unsafe_allow_html=True)
            else:
                st.warning("No movies found. Try a different search term.")
    
    # Popular Movies
    elif page == "⭐ Popular":
        st.markdown('<div class="section-header">⭐ Most Popular Movies</div>', unsafe_allow_html=True)
        
        n = st.slider("Number of movies", 10, 50, 20, key="popular_n")
        
        if st.button("Show Popular Movies", key="show_popular"):
            popular = ratings.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
            popular = popular[popular['count'] >= 10]
            popular = popular.sort_values('mean', ascending=False).head(n)
            
            popular_with_titles = popular.merge(movies, on='movie_id')
            
            # Display in grid
            cols = st.columns(5)
            for idx, (_, row) in enumerate(popular_with_titles.iterrows()):
                with cols[idx % 5]:
                    st.markdown(create_movie_card(
                        {'title': row['title'], 'genres': row.get('genres', '')},
                        rating=row['mean']
                    ), unsafe_allow_html=True)
    
    # Similar Movies
    elif page == "🎭 Similar Movies":
        st.markdown('<div class="section-header">🎭 Find Similar Movies</div>', unsafe_allow_html=True)
        
        # Movie search
        movie_search = st.text_input("", placeholder="Search for a movie...", key="similar_search")
        
        if movie_search:
            matching_movies = movies[movies['title'].str.contains(movie_search, case=False, na=False)]
            
            if len(matching_movies) > 0:
                selected_movie = st.selectbox(
                    "Select a movie",
                    matching_movies['title'].tolist(),
                    key="select_movie"
                )
                movie_id = matching_movies[matching_movies['title'] == selected_movie]['movie_id'].iloc[0]
                
                n_recs = st.slider("Number of similar movies", 5, 20, 10, key="n_similar")
                
                if st.button("🎬 Find Similar Movies", key="find_similar"):
                    if 'content_based' in models:
                        with st.spinner("🎭 Finding similar movies..."):
                            recommendations = models['content_based'].recommend_similar_movies(
                                movie_id, n_recommendations=n_recs
                            )
                        
                        st.success(f"✨ Found {len(recommendations)} similar movies!")
                        
                        # Display selected movie
                        selected_movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
                        st.markdown(f'<div class="section-header">Similar to: {selected_movie_info["title"]}</div>', unsafe_allow_html=True)
                        
                        # Display recommendations in grid
                        cols = st.columns(5)
                        for idx, rec in enumerate(recommendations):
                            with cols[idx % 5]:
                                st.markdown(create_movie_card(
                                    {'title': rec['title'], 'genres': rec.get('genres', 'N/A')},
                                    similarity=rec['similarity_score']
                                ), unsafe_allow_html=True)
                    else:
                        st.error("⚠️ Content-based model not available.")
            else:
                st.warning("No movies found. Try a different search term.")

if __name__ == "__main__":
    main()
