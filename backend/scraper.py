import feedparser
import httpx
from bs4 import BeautifulSoup
import asyncio

async def get_recent_watches(username: str):
    """Fetches the last 20 watched/reviewed movies from RSS."""
    rss_url = f"https://letterboxd.com/{username}/rss/"
    
    # Run synchronous feedparser in a separate thread
    feedparser.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    feed = await asyncio.to_thread(feedparser.parse, rss_url)
    
    recent_movies = []
    for entry in feed.entries[:10]: # Limit to 10 for recent watches
        # Extract title from "Title - Rating" format if rating exists
        title = entry.title
        rating = "N/A"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0]
            rating = parts[1]
        
        recent_movies.append({
            "title": title,
            "rating": rating,
            "link": entry.link
        })
    return recent_movies

async def get_watchlist(username: str, max_pages: int = 10):
    watchlist_movies = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        for page in range(1, max_pages + 1):
            url = f"https://letterboxd.com/{username}/watchlist/page/{page}/"
            response = await client.get(url)
            
            if response.status_code == 404:
                if page == 1:
                    return None
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            posters = soup.select('[data-film-slug], [data-item-slug]')
            
            if not posters:
                break # No more movies on this page
                
            for poster in posters:
                slug = poster.get('data-film-slug') or poster.get('data-item-slug', '')
                img = poster.find('img')
                title = img.get('alt', slug.replace('-', ' ').title()) if img else slug.replace('-', ' ').title()
                
                if slug:
                    watchlist_movies.append({
                        "title": title,
                        "link": f"https://letterboxd.com/film/{slug}/"
                    })
                    
    return watchlist_movies
