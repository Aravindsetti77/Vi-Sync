import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from cache import get_cache, set_cache

# Initialize the model lazily to avoid heavy loading on startup if not immediately used
_model = None

def get_model():
    global _model
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
        "all-MiniLM-L6-v2",        # Use .safetensors version
        torch_dtype=torch.float16, # Half-precision (saves 50% RAM/Time)
        low_cpu_mem_usage=True,    # Instant memory-mapping (mmap)
        device_map="auto"          # Smartest hardware placement
    )
    return _model

async def get_recommendation(recent_watches, watchlist, skipped_links=None, cache_key=None):
    if skipped_links is None:
        skipped_links = []
        
    # Check cache if a cache key is provided
    cached_data = None
    if cache_key:
        cached_data = await get_cache(cache_key)

    if cached_data:
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

        # Create a unified string for each recent watch
        recent_texts = []
        for m in recent_watches:
            text = m.get('title', '')
            if m.get('rating') and m['rating'] != "N/A":
                text += f" (Rating: {m['rating']}) (Liked: {'Yes' if m.get('liked') else 'No'})"
            if m.get('directors'):
                text += f" Director: {', '.join(m['directors'])}."
            if m.get('genres'):
                text += f" Genres: {', '.join(m['genres'])}."
            if m.get('description'):
                text += f"\nDescription/Review: {m['description']}"
            recent_texts.append(text)
            
        recent_embeddings = model.encode(recent_texts)
        user_vector = np.mean(recent_embeddings, axis=0).reshape(1, -1)
        
        watchlist_texts = []
        for m in watchlist:
            text = m.get('title', '')
            if m.get('directors'):
                text += f" Director: {', '.join(m['directors'])}."
            if m.get('genres'):
                text += f" Genres: {', '.join(m['genres'])}."
            if m.get('description'):
                text += f"\nDescription/Review: {m['description']}"
            watchlist_texts.append(text)
            
        watchlist_embeddings = model.encode(watchlist_texts)

        if cache_key:
            await set_cache(cache_key, {
                'user_vector': user_vector,
                'watchlist_embeddings': watchlist_embeddings,
                'watchlist': watchlist
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
