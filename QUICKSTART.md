# Quick Start Guide

## 🚀 Setup in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset

```bash
python src/data_loader.py
```

This will automatically download the MovieLens 100K dataset.

### Step 3: Train Models

```bash
# Train collaborative filtering
python src/models/collaborative.py

# Train content-based filtering
python src/models/content_based.py

# Train hybrid model
python src/models/hybrid.py
```

### Step 4: Run API

```bash
python start_api.py
```

Visit: http://localhost:8000/docs

### Step 5: Run Dashboard

```bash
streamlit run dashboard/app.py
```

Visit: http://localhost:8501

## 📊 What You Get

- **100,000 ratings** from 943 users
- **1,682 movies** with metadata
- **3 recommendation algorithms**
- **REST API** for integration
- **Interactive dashboard** for exploration

## 🎯 Example Usage

### Get Recommendations for User

```python
from src.models.collaborative import CollaborativeFiltering

cf = CollaborativeFiltering()
cf.load_model('data/models/collaborative_svd.pkl')

recommendations = cf.recommend_for_user(user_id=1, n_recommendations=10)
```

### Find Similar Movies

```python
from src.models.content_based import ContentBasedFiltering

cb = ContentBasedFiltering()
cb.load_model('data/models/content_based_model.pkl')

similar = cb.recommend_similar_movies(movie_id=1, n_recommendations=10)
```

## 🐳 Docker

```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

## 📝 Next Steps

1. Explore the dashboard
2. Try different recommendation algorithms
3. Customize for your use case
4. Deploy to production

Enjoy your movie recommender! 🎬

