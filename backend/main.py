from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from scraper import get_recent_watches, get_watchlist
from recommender import get_recommendation
import os
import asyncio

app = FastAPI(title="Letterboxd Curation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    username_or_url: str

class GroupRecommendRequest(BaseModel):
    usernames: List[str]

def extract_username(input_str: str) -> str:
    input_str = input_str.strip().rstrip('/')
    if "letterboxd.com/" in input_str:
        return input_str.split("letterboxd.com/")[-1].split('/')[0]
    return input_str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    username = extract_username(request.username_or_url)
    if not username:
        raise HTTPException(status_code=400, detail="Invalid username or URL")
        
    try:
        recent_watches = await get_recent_watches(username)
        watchlist = await get_watchlist(username)
        
        if watchlist is None:
            raise HTTPException(status_code=404, detail="Watchlist not found or empty.")
            
        recommendation = await get_recommendation(recent_watches, watchlist)
        
        return {
            "username": username,
            "recommendation": recommendation,
            "stats": {
                "recent_watches_count": len(recent_watches),
                "watchlist_count": len(watchlist)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_recommend")
async def group_recommend(request: GroupRecommendRequest):
    usernames = [extract_username(u) for u in request.usernames if extract_username(u)]
    if len(usernames) < 2:
        raise HTTPException(status_code=400, detail="At least two valid usernames are required for a group match.")
        
    try:
        async def fetch_user_data(user):
            rw = await get_recent_watches(user)
            wl = await get_watchlist(user)
            return rw, wl
            
        results = await asyncio.gather(*[fetch_user_data(u) for u in usernames])
        
        all_recent_watches = []
        all_watchlists = []
        
        for rw, wl in results:
            if rw:
                all_recent_watches.extend(rw)
            if wl:
                all_watchlists.extend(wl)
                
        if not all_watchlists:
            raise HTTPException(status_code=404, detail="No watchlists found or all watchlists are empty.")
            
        seen_links = set()
        unique_watchlist = []
        for m in all_watchlists:
            if m['link'] not in seen_links:
                seen_links.add(m['link'])
                unique_watchlist.append(m)
                
        recommendation = await get_recommendation(all_recent_watches, unique_watchlist)
        
        return {
            "usernames": usernames,
            "recommendation": recommendation,
            "stats": {
                "recent_watches_count": len(all_recent_watches),
                "watchlist_count": len(unique_watchlist)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
