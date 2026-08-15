import feedparser
import httpx
from bs4 import BeautifulSoup
import asyncio
import datetime

async def get_movie_metadata(client, url, semaphore):
    async with semaphore:
        try:
            res = await client.get(url, follow_redirects=True)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                genres = list(dict.fromkeys([a.text for a in soup.find_all('a') if '/films/genre/' in a.get('href', '')]))
                directors = list(dict.fromkeys([a.text for a in soup.find_all('a') if '/director/' in a.get('href', '')]))
                poster_url = None
                script_tag = soup.find('script', {'type': 'application/ld+json'})
                if script_tag:
                    try:
                        import json
                        text = script_tag.text.strip()
                        if text.startswith('/* <![CDATA[ */'):
                            text = text.replace('/* <![CDATA[ */', '', 1)
                        if text.endswith('/* ]]> */'):
                            text = text.rsplit('/* ]]> */', 1)[0]
                        data = json.loads(text)
                        if 'image' in data:
                            # 'image' might be a string or list, typically string
                            if isinstance(data['image'], list) and len(data['image']) > 0:
                                poster_url = data['image'][0]
                            elif isinstance(data['image'], str):
                                poster_url = data['image']
                    except Exception:
                        pass
                return {"genres": genres, "directors": directors, "poster": poster_url}
        except Exception:
            pass
        return {"genres": [], "directors": [], "poster": None}

async def get_recent_watches(username: str):
    """Fetches the last 20 watched/reviewed movies from RSS."""
    rss_url = f"https://letterboxd.com/{username}/rss/"
    
    # Run synchronous feedparser in a separate thread
    feedparser.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    feed = await asyncio.to_thread(feedparser.parse, rss_url)
    
    recent_movies = []
    liked_movies = []
    
    current_year = datetime.datetime.now().year
    
    for entry in feed.entries:
        year_str = entry.get('letterboxd_filmyear', '')
        if year_str.isdigit() and int(year_str) > current_year:
            continue
            
        title = entry.title
        rating = "N/A"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0]
            rating = parts[1]
        
        raw_description = entry.get('description', entry.get('summary', ''))
        description = BeautifulSoup(raw_description, 'html.parser').get_text(separator=' ', strip=True) if raw_description else ""
        
        movie = {
            "title": title,
            "rating": rating,
            "link": entry.link,
            "liked": entry.get('letterboxd_memberlike') == 'Yes',
            "description": description
        }
        
        recent_movies.append(movie)
        if movie["liked"]:
            liked_movies.append(movie)
            
    # Determine the subset to process (max 20)
    selected_movies = liked_movies[:20] if liked_movies else recent_movies[:20]
    
    # Fetch metadata (genres, directors)
    semaphore = asyncio.Semaphore(5)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        tasks = [get_movie_metadata(client, m["link"], semaphore) for m in selected_movies]
        metadata_results = await asyncio.gather(*tasks)
        for m, meta in zip(selected_movies, metadata_results):
            m["genres"] = meta["genres"]
            m["directors"] = meta["directors"]
            m["poster"] = meta.get("poster")
            
    return selected_movies

async def get_watchlist(username: str, max_pages: int = 3):
    watchlist_movies = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    async def fetch_page(client, page):
        url = f"https://letterboxd.com/{username}/watchlist/page/{page}/"
        response = await client.get(url)
        if response.status_code == 404:
            return page, None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        posters = soup.select('[data-film-slug], [data-item-slug]')
        
        if not posters:
            return page, []
            
        movies = []
        current_year = datetime.datetime.now().year
        
        for poster in posters:
            name = poster.get('data-item-name', '') or poster.get('data-film-name', '')
            unreleased = False
            if name.endswith(')'):
                try:
                    year_str = name.split('(')[-1].replace(')', '')
                    if year_str.isdigit() and int(year_str) > current_year:
                        unreleased = True
                except:
                    pass
            
            if unreleased:
                continue

            slug = poster.get('data-film-slug') or poster.get('data-item-slug', '')
            img = poster.find('img')
            title = img.get('alt', slug.replace('-', ' ').title()) if img else slug.replace('-', ' ').title()
            
            if slug:
                movies.append({
                    "title": title,
                    "link": f"https://letterboxd.com/film/{slug}/"
                })
        return page, movies

    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        # Fetch page 1 first to check if watchlist exists
        page1_res = await fetch_page(client, 1)
        if page1_res[1] is None:
            return None
        
        watchlist_movies.extend(page1_res[1])
        
        # If page 1 was empty or didn't have a full page (usually 28 movies), we could stop,
        # but to be safe and simple, we fetch the rest concurrently.
        if len(page1_res[1]) > 0:
            tasks = [fetch_page(client, page) for page in range(2, max_pages + 1)]
            results = await asyncio.gather(*tasks)
            
            results.sort(key=lambda x: x[0])
            for page, movies in results:
                if movies is None or not movies:
                    break
                watchlist_movies.extend(movies)
                
        # Fetch metadata (genres, directors) for watchlist movies
        semaphore = asyncio.Semaphore(5)
        tasks = [get_movie_metadata(client, m["link"], semaphore) for m in watchlist_movies]
        metadata_results = await asyncio.gather(*tasks)
        for m, meta in zip(watchlist_movies, metadata_results):
            m["genres"] = meta["genres"]
            m["directors"] = meta["directors"]
            m["poster"] = meta.get("poster")
                    
    return watchlist_movies
