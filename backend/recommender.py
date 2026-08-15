import numpy as np
import hashlib
import asyncio
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from cache import get_cache, set_cache, delete_cache

# Initialize the model lazily to avoid heavy loading on startup if not immediately used
_model = None

def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=1)
    return _model

def warmup_model():
    """Pre-load the model and run a dummy encode to warm up all internal caches.
    Call this at application startup so the first real request is fast."""
    model = get_model()
    list(model.embed(["warmup"]))

def _compute_fingerprint(recent_watches, watchlist):
    """Create a hash fingerprint from the current watchlist and recent watches.
    If the user updates their watchlist or diary on Letterboxd, this fingerprint
    will change, which triggers a cache invalidation and model re-encoding."""
    parts = []
    # Include recent watch links and ratings (diary changes)
    if recent_watches:
        for m in recent_watches:
            parts.append(f"r:{m.get('link', '')}:{m.get('rating', '')}:{m.get('liked', '')}")
    # Include watchlist links (watchlist additions/removals)
    if watchlist:
        for m in watchlist:
            parts.append(f"w:{m.get('link', '')}")
    raw = "|".join(sorted(parts))
    return hashlib.sha256(raw.encode()).hexdigest()

def _build_text(movie, include_rating=False):
    """Build a single text representation for a movie for embedding."""
    text = movie.get('title', '')
    if include_rating:
        if movie.get('rating') and movie['rating'] != "N/A":
            text += f" (Rating: {movie['rating']}) (Liked: {'Yes' if movie.get('liked') else 'No'})"
    if movie.get('directors'):
        text += f" Director: {', '.join(movie['directors'])}."
    if movie.get('genres'):
        text += f" Genres: {', '.join(movie['genres'])}."
    if movie.get('description'):
        text += f"\nDescription/Review: {movie['description']}"
    return text

async def get_recommendation(recent_watches, watchlist, skipped_links=None, cache_key=None):
    if skipped_links is None:
        skipped_links = []
        
    # Check cache if a cache key is provided
    cached_data = None
    use_cache = False
    
    if cache_key:
        cached_data = await get_cache(cache_key)

    if cached_data and recent_watches is not None and watchlist is not None:
        # We have fresh data AND a cache — check if the data has changed
        current_fingerprint = _compute_fingerprint(recent_watches, watchlist)
        cached_fingerprint = cached_data.get('fingerprint', '')
        
        if current_fingerprint == cached_fingerprint:
            # Data hasn't changed, safe to use cache
            use_cache = True
        else:
            # Data changed (user updated watchlist/diary) — invalidate cache
            await delete_cache(cache_key)
            cached_data = None
            use_cache = False
    elif cached_data and recent_watches is None and watchlist is None:
        # Called from /skip endpoint — no fresh data available, use cache
        use_cache = True

    if use_cache and cached_data:
        # Restore from cache
        user_vector = np.array(cached_data['user_vector'])
        watchlist_embeddings = np.array(cached_data['watchlist_embeddings'])
        watchlist = cached_data['watchlist']
    else:
        if not watchlist:
            return None
        if not recent_watches:
            import random
            # Filter out skipped movies
            valid_watchlist = [m for m in watchlist if m.get('link') not in skipped_links]
            if not valid_watchlist:
                return None
            choice = random.choice(valid_watchlist)
            choice['reason'] = "Random selection since no recent watches were found."
            return choice

        model = get_model()

        # Build text representations
        recent_texts = [_build_text(m, include_rating=True) for m in recent_watches]
        watchlist_texts = [_build_text(m, include_rating=False) for m in watchlist]

        # Single batched encode — much faster than two separate encode() calls
        # because it avoids re-initializing internal tokenizer/model state twice
        all_texts = recent_texts + watchlist_texts
        
        def run_embedding():
            # Reduced batch size to 4 to prevent Out of Memory (OOM) crashes on 512MB instances
            return np.array(list(model.embed(all_texts, batch_size=4)))
            
        all_embeddings = await asyncio.to_thread(run_embedding)

        recent_embeddings = all_embeddings[:len(recent_texts)]
        watchlist_embeddings = all_embeddings[len(recent_texts):]

        user_vector = np.mean(recent_embeddings, axis=0).reshape(1, -1)

        if cache_key:
            fingerprint = _compute_fingerprint(recent_watches, watchlist)
            await set_cache(cache_key, {
                'user_vector': user_vector,
                'watchlist_embeddings': watchlist_embeddings,
                'watchlist': watchlist,
                'fingerprint': fingerprint
            })
        
    # Compute cosine similarities
    similarities = cosine_similarity(user_vector, watchlist_embeddings)[0]
    
    # Filter out skipped movies
    # Set similarity of skipped movies to -1 (lowest possible)
    for i, m in enumerate(watchlist):
        if m.get('link') in skipped_links:
            similarities[i] = -1.0
            
    best_index = np.argmax(similarities)
    
    # If the max similarity is -1, it means all movies are skipped
    if similarities[best_index] == -1.0:
        return None # No movies left to recommend
        
    best_score = similarities[best_index]
    best_match = watchlist[best_index]
    
    match_percentage = int(best_score * 100)
    best_match['reason'] = f"Vibe match! Selected via semantic similarity ({match_percentage}% match) based on your recent activity."
    
    return best_match
