import numpy as np
import hashlib
import asyncio
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from cache import get_cache, set_cache, delete_cache

# Initialize the model lazily to avoid heavy loading on startup if not immediately used
_model = None

# If you convert your custom model using the export.py script, point this path to the ONNX folder
LOCAL_MODEL_PATH = "onnx_model" 
MODEL_CACHE_DIR = "local_model_cache"

def get_model():
    global _model
    if _model is None:
        import os
        
        # Check if a custom ONNX converted model exists locally
        if os.path.exists(LOCAL_MODEL_PATH) and os.path.isdir(LOCAL_MODEL_PATH):
            from fastembed.common.model_description import ModelSource, PoolingType
            model_name = "custom-local-model"
            
            try:
                TextEmbedding.add_custom_model(
                    model=model_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=""), 
                    dim=384, # Adjust to your local model's embedding dimension
                    model_file="model.onnx",
                )
            except ValueError:
                pass # Already registered
                
            _model = TextEmbedding(model_name=model_name, cache_dir=LOCAL_MODEL_PATH, threads=1)
        else:
            # Fallback to the standard model, but cached locally to prevent startup downloads
            cache_dir = MODEL_CACHE_DIR if os.path.exists(MODEL_CACHE_DIR) else None
            _model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=cache_dir, threads=1)
            
    return _model

def warmup_model():
    """Pre-load the model and run a dummy encode to warm up all internal caches.
    Call this at application startup so the first real request is fast."""
    model = get_model()
    list(model.embed(["warmup"]))

def _compute_fingerprint(user_state):
    """Create a hash fingerprint from the user state."""
    if not user_state:
        return "default"
    
    parts = []
    if "rss" in user_state:
        parts.extend([f"r:{link}" for link in user_state["rss"]])
    if "watchlist" in user_state:
        parts.extend([f"w:{link}" for link in user_state["watchlist"]])
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

async def get_recommendation(recent_watches, watchlist, skipped_links=None, cache_key=None, current_fingerprint=None):
    if skipped_links is None:
        skipped_links = []
        
    cached_data = None
    use_cache = False
    
    if cache_key:
        cached_data = await get_cache(cache_key)

    if cached_data and current_fingerprint is not None:
        cached_fingerprint = cached_data.get('fingerprint', '')
        if current_fingerprint == cached_fingerprint:
            use_cache = True
        else:
            await delete_cache(cache_key)
            cached_data = None
            use_cache = False
    elif cached_data and current_fingerprint is None and recent_watches is None and watchlist is None:
        # Called from /skip endpoint — no fresh data available, use cache
        use_cache = True

    stats = {"recent_watches_count": 0, "watchlist_count": 0}

    if use_cache and cached_data:
        # Restore from cache
        user_vector = np.array(cached_data['user_vector'])
        watchlist_embeddings = np.array(cached_data['watchlist_embeddings'])
        watchlist = cached_data['watchlist']
        stats = cached_data.get('stats', stats)
    else:
        if not watchlist:
            return None, stats
            
        stats["recent_watches_count"] = len(recent_watches) if recent_watches else 0
        stats["watchlist_count"] = len(watchlist)
        
        if not recent_watches:
            import random
            valid_watchlist = [m for m in watchlist if m.get('link') not in skipped_links]
            if not valid_watchlist:
                return None, stats
            choice = random.choice(valid_watchlist)
            choice['reason'] = "Random selection since no recent watches were found."
            return choice, stats

        model = get_model()
        recent_texts = [_build_text(m, include_rating=True) for m in recent_watches]
        watchlist_texts = [_build_text(m, include_rating=False) for m in watchlist]
        all_texts = recent_texts + watchlist_texts
        
        def run_embedding():
            return np.array(list(model.embed(all_texts, batch_size=4)))
            
        all_embeddings = await asyncio.to_thread(run_embedding)

        recent_embeddings = all_embeddings[:len(recent_texts)]
        watchlist_embeddings = all_embeddings[len(recent_texts):]
        user_vector = np.mean(recent_embeddings, axis=0).reshape(1, -1)

        if cache_key and current_fingerprint:
            await set_cache(cache_key, {
                'user_vector': user_vector.tolist(),
                'watchlist_embeddings': watchlist_embeddings.tolist(),
                'watchlist': watchlist,
                'fingerprint': current_fingerprint,
                'stats': stats
            })
        
    # Compute cosine similarities
    similarities = cosine_similarity(user_vector, watchlist_embeddings)[0]
    
    for i, m in enumerate(watchlist):
        if m.get('link') in skipped_links:
            similarities[i] = -1.0
            
    best_index = np.argmax(similarities)
    
    if similarities[best_index] == -1.0:
        return None, stats
        
    best_score = similarities[best_index]
    best_match = watchlist[best_index]
    match_percentage = int(best_score * 100)
    best_match['reason'] = f"Vibe match! Selected via semantic similarity ({match_percentage}% match) based on your recent activity."
    
    return best_match, stats
