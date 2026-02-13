# 🎬 Movie Recommender System

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## 🌐 Live Demo

**🎬 [Try the Netflix-Style Dashboard →](https://znwqdxdrscbslsippw3mrp.streamlit.app/)**

Experience the movie recommender with a beautiful Netflix-style interface!

A comprehensive movie recommendation system with a **Netflix-style dashboard** using collaborative filtering and content-based filtering algorithms. Built with Python, FastAPI, and Streamlit.

## 🎯 Features

- **Multiple Recommendation Algorithms**:
  - Collaborative Filtering (NMF-based)
  - Content-Based Filtering
  - Hybrid Approach
- **Netflix-Style Dashboard**: Beautiful dark-themed UI with movie cards
- **REST API**: FastAPI-based API for recommendations
- **Real Dataset**: MovieLens 100K dataset (100,000 ratings)
- **Performance Metrics**: RMSE, similarity scores
- **Docker Support**: Easy deployment

## 📊 Dataset

Uses the **MovieLens 100K Dataset**:
- 100,000 ratings from 943 users on 1,682 movies
- User demographics
- Movie metadata (genres, release dates)

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas & NumPy**: Data manipulation
- **scikit-learn**: Machine learning (NMF for collaborative filtering)
- **FastAPI**: REST API
- **Streamlit**: Netflix-style web dashboard
- **Docker**: Containerization

## 📁 Project Structure

```
movie-recommender-system/
├── README.md
├── requirements.txt
├── Dockerfile
├── start_api.py
├── data/
│   ├── raw/              # MovieLens dataset
│   ├── processed/        # Processed data
│   └── models/          # Trained models
├── src/
│   ├── data_loader.py    # Load MovieLens data
│   ├── preprocessing.py  # Data preprocessing
│   ├── models/
│   │   ├── collaborative_simple.py  # NMF-based CF
│   │   ├── content_based.py          # Content-based
│   │   └── hybrid.py                 # Hybrid approach
│   ├── evaluation.py     # Model evaluation
│   └── visualization.py  # Data visualization
├── api/
│   └── main.py           # FastAPI application
├── dashboard/
│   └── app.py            # Netflix-style Streamlit dashboard
└── .streamlit/
    └── config.toml       # Streamlit theme config
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/movie-recommender-system.git
cd movie-recommender-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Automatically downloads MovieLens 100K dataset
python src/data_loader.py
```

### Train Models

```bash
# Train collaborative filtering
python src/models/collaborative_simple.py

# Train content-based filtering
python src/models/content_based.py
```

### Run API

```bash
python start_api.py
```

Visit: http://localhost:8000/docs

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

**Or try the live version**: [🌐 Live Dashboard](https://znwqdxdrscbslsippw3mrp.streamlit.app/)

## 🎨 Netflix-Style Dashboard

The dashboard features:
- **Dark Theme**: Netflix-inspired black background
- **Movie Cards**: Hover effects and smooth animations
- **Horizontal Scrolling**: Movie rows like Netflix
- **Multiple Pages**: Home, For You, Discover, Popular, Similar Movies
- **Responsive Design**: Works on all screen sizes

## 📈 Model Performance

- **Collaborative Filtering RMSE**: ~2.06
- **Content-Based Similarity**: Cosine similarity on genres
- **Hybrid Approach**: Combines both methods

## 🔍 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /recommend/user/{user_id}` - Get recommendations for user
- `GET /recommend/movie/{movie_id}` - Get similar movies
- `GET /movies/popular` - Get popular movies
- `GET /movies/{movie_id}` - Get movie details

## 🐳 Docker Deployment

```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

## 📝 Example Usage

### Python API Client

```python
import requests

# Get recommendations for user 1
response = requests.get("http://localhost:8000/recommend/user/1?n_recommendations=10")
recommendations = response.json()
print(recommendations)
```

### Dashboard

1. Run: `streamlit run dashboard/app.py`
2. Open: http://localhost:8501
3. Navigate through pages:
   - **Home**: Trending movies
   - **For You**: Personalized recommendations
   - **Discover**: Search movies
   - **Popular**: Top rated movies
   - **Similar Movies**: Find similar movies

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Recommendation systems (collaborative & content-based)
- ✅ Real dataset usage (MovieLens)
- ✅ REST API development (FastAPI)
- ✅ Interactive dashboard (Streamlit)
- ✅ Modern UI design (Netflix-style)
- ✅ Data science workflow
- ✅ Production-ready code

## 📝 License

MIT License

## 👤 Author

Created as a portfolio project demonstrating recommendation systems and machine learning.

## 🙏 Acknowledgments

- MovieLens dataset from GroupLens Research
- FastAPI and Streamlit communities

---

⭐ **If you find this project helpful, please give it a star!** ⭐
