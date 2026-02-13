#!/usr/bin/env python3
"""Start Movie Recommender API"""

import uvicorn

if __name__ == "__main__":
    print("🎬 Starting Movie Recommender API...")
    print("📍 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

