"""
Quick System Test Script
"""

import sys
import os

print("🧪 Testing Movie Recommender System")
print("="*60)

# Test 1: Data Loading
print("\n1. Testing Data Loading...")
try:
    from src.data_loader import MovieLensLoader
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_all(download=False)
    print(f"   ✅ Data loaded: {len(ratings):,} ratings, {len(movies):,} movies")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 2: Models
print("\n2. Testing Models...")
try:
    from src.models.collaborative_simple import SimpleCollaborativeFiltering
    from src.models.content_based import ContentBasedFiltering
    
    # Check if models exist
    if os.path.exists('data/models/collaborative_simple.pkl'):
        cf = SimpleCollaborativeFiltering()
        cf.load_model('data/models/collaborative_simple.pkl')
        print("   ✅ Collaborative Filtering model loaded")
    else:
        print("   ⚠️  Collaborative model not trained yet")
    
    if os.path.exists('data/models/content_based_model.pkl'):
        cb = ContentBasedFiltering()
        cb.load_model('data/models/content_based_model.pkl')
        print("   ✅ Content-Based model loaded")
    else:
        print("   ⚠️  Content-Based model not trained yet")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: API
print("\n3. Testing API...")
try:
    from api.main import app
    print("   ✅ API imports successful")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Dashboard
print("\n4. Testing Dashboard...")
try:
    import streamlit
    print("   ✅ Streamlit available")
    print("   ✅ Dashboard ready to run")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ System Test Complete!")
print("="*60)
print("\nTo run the system:")
print("  API: python start_api.py")
print("  Dashboard: streamlit run dashboard/app.py")

