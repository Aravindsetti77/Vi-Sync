import os
import asyncio
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from scraper import get_recent_watches, get_watchlist, get_user_state
from recommender import get_recommendation, warmup_model, _compute_fingerprint
from database import init_db, AsyncSessionLocal, SkippedMovie
from cache import close_cache, check_cache_health, get_cache, set_cache
from sqlalchemy.future import select
from sqlalchemy import delete
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Retry database connection (Wait for DB to boot on Render)
    for _ in range(12):
        try:
            await init_db()
            logging.info("Database initialized successfully.")
            break
        except Exception as e:
            logging.warning(f"Database connection failed, retrying in 5s... ({e})")
            await asyncio.sleep(5)
    else:
        logging.error("Failed to connect to database after 60 seconds.")
        
    # Retry Redis connection
    for _ in range(12):
        cache_ok = await check_cache_health()
        if cache_ok:
            logging.info("Redis cache connected successfully.")
            break
        logging.warning("Redis cache connection failed, retrying in 5s...")
        await asyncio.sleep(5)
    else:
        logging.error("Failed to connect to Redis cache after 60 seconds.")
        
    # Pre-load the fastembed model at startup
    try:
        await asyncio.to_thread(warmup_model)
        logging.info("Model warmed up successfully.")
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")

    yield
    await close_cache()

app = FastAPI(title="Letterboxd Curation Engine", lifespan=lifespan)

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

class SkipRequest(BaseModel):
    usernames: List[str]
    movie_link: str

class UnskipRequest(BaseModel):
    usernames: List[str]
    movie_link: str

def extract_username(input_str: str) -> str:
    input_str = input_str.strip().rstrip('/')
    if "letterboxd.com/" in input_str:
        return input_str.split("letterboxd.com/")[-1].split('/')[0].lower()
    return input_str.lower()

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

async def get_skipped_links(ip_address: str, cache_key_id: str) -> List[str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SkippedMovie.movie_link).where(
                SkippedMovie.ip_address == ip_address,
                SkippedMovie.username == cache_key_id
            )
        )
        return [row[0] for row in result.all()]

async def clear_skipped_links(ip_address: str, cache_key_id: str):
    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(SkippedMovie).where(
                SkippedMovie.ip_address == ip_address,
                SkippedMovie.username == cache_key_id
            )
        )
        await session.commit()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/avatar/{username}")
async def get_avatar(username: str):
    """Attempt to scrape the user's avatar. Returns a placeholder if blocked by Cloudflare."""
    cache_key = f"avatar_{username}"
    cached = await get_cache(cache_key)
    if cached:
        return {"avatar": cached}
        
    url = f"https://letterboxd.com/{username}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    avatar_url = None  # None means no real avatar found; frontend keeps its SVG placeholder
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=5.0) as client:
            res = await client.get(url)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                og_image = soup.find('meta', property='og:image')
                if og_image and 'default-share' not in og_image.get('content', ''):
                    avatar_url = og_image['content']
                else:
                    img = soup.find('img', alt=lambda x: x and 'avatar' in x.lower())
                    if img and img.get('src'):
                        avatar_url = img['src']
    except Exception:
        pass
        
    # Cache for 24 hours (even a None result, to avoid repeated failed fetches)
    await set_cache(cache_key, avatar_url or "", 86400)
    return {"avatar": avatar_url}

@app.post("/recommend")
async def recommend(request: RecommendRequest, req: Request):
    username = extract_username(request.username_or_url)
    if not username:
        raise HTTPException(status_code=400, detail="Invalid username or URL")
        
    ip_address = get_client_ip(req)
    cache_key = f"user_{username}"
    
    try:
        await clear_skipped_links(ip_address, cache_key)
        skipped_links = []
        
        user_state = await get_user_state(username)
        if not user_state:
            raise HTTPException(status_code=404, detail="Watchlist not found or empty.")
            
        cached_data = await get_cache(cache_key)
        current_fingerprint = _compute_fingerprint(user_state)
        
        if cached_data and cached_data.get('fingerprint') == current_fingerprint:
            recent_watches, watchlist = None, None
        else:
            recent_watches, watchlist = await asyncio.gather(
                get_recent_watches(username),
                get_watchlist(username)
            )
            
        recommendation, stats = await get_recommendation(recent_watches, watchlist, skipped_links, cache_key, current_fingerprint)
        
        return {
            "username": username,
            "usernames": [username],
            "recommendation": recommendation,
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_recommend")
async def group_recommend(request: GroupRecommendRequest, req: Request):
    usernames = [extract_username(u) for u in request.usernames if extract_username(u)]
    if len(usernames) < 2:
        raise HTTPException(status_code=400, detail="At least two valid usernames are required for a group match.")
        
    ip_address = get_client_ip(req)
    cache_key = f"group_{'_'.join(sorted(usernames))}"
    
    try:
        await clear_skipped_links(ip_address, cache_key)
        skipped_links = []
        
        states = await asyncio.gather(*[get_user_state(u) for u in usernames])
        combined_state = {"rss": [], "watchlist": []}
        for st in states:
            if st:
                combined_state["rss"].extend(st["rss"])
                combined_state["watchlist"].extend(st["watchlist"])
                
        if not combined_state["watchlist"]:
            raise HTTPException(status_code=404, detail="No watchlists found or all watchlists are empty.")
            
        cached_data = await get_cache(cache_key)
        current_fingerprint = _compute_fingerprint(combined_state)
        
        if cached_data and cached_data.get('fingerprint') == current_fingerprint:
            all_recent_watches, unique_watchlist = None, None
        else:
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
                    
            seen_links = set()
            unique_watchlist = []
            for m in all_watchlists:
                if m['link'] not in seen_links:
                    seen_links.add(m['link'])
                    unique_watchlist.append(m)
                    
        recommendation, stats = await get_recommendation(all_recent_watches, unique_watchlist, skipped_links, cache_key, current_fingerprint)
        
        return {
            "usernames": usernames,
            "recommendation": recommendation,
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/skip")
async def skip_movie(request: SkipRequest, req: Request):
    ip_address = get_client_ip(req)
    if len(request.usernames) == 1:
        cache_key_id = f"user_{request.usernames[0]}"
    else:
        cache_key_id = f"group_{'_'.join(sorted(request.usernames))}"
        
    # Save skip to PostgreSQL
    async with AsyncSessionLocal() as session:
        skip_entry = SkippedMovie(ip_address=ip_address, username=cache_key_id, movie_link=request.movie_link)
        session.add(skip_entry)
        await session.commit()
        
    skipped_links = await get_skipped_links(ip_address, cache_key_id)
    
    recommendation, _ = await get_recommendation(None, None, skipped_links=skipped_links, cache_key=cache_key_id)
    
    if not recommendation:
        raise HTTPException(status_code=404, detail="No more movies to recommend or cache expired. Please search again.")
        
    return {
        "usernames": request.usernames,
        "recommendation": recommendation
    }

@app.post("/unskip")
async def unskip_movie(request: UnskipRequest, req: Request):
    ip_address = get_client_ip(req)
    if len(request.usernames) == 1:
        cache_key_id = f"user_{request.usernames[0]}"
    else:
        cache_key_id = f"group_{'_'.join(sorted(request.usernames))}"
        
    # Remove skip from PostgreSQL
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SkippedMovie).where(
                SkippedMovie.ip_address == ip_address,
                SkippedMovie.username == cache_key_id,
                SkippedMovie.movie_link == request.movie_link
            )
        )
        entry = result.scalar_one_or_none()
        if entry:
            await session.delete(entry)
            await session.commit()
            
    return {"status": "success"}
