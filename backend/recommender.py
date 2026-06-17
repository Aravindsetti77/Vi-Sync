import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the model lazily to avoid heavy loading on startup if not immediately used
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

async def get_recommendation(recent_watches, watchlist):
    if not watchlist:
        return None
        
    if not recent_watches:
        import random
        choice = random.choice(watchlist)
        choice['reason'] = "Random selection since no recent watches were found."
        return choice

    model = get_model()

    # Create a unified string for each recent watch
    recent_texts = []
    for m in recent_watches:
        text = m.get('title', '')
        if m.get('rating') and m['rating'] != "N/A":
            # If we had descriptions we would add them here. For now, titles and ratings.
            text += f" (Rating: {m['rating']})"
        recent_texts.append(text)
        
    # Embed the recent watches
    recent_embeddings = model.encode(recent_texts)
    
    # Calculate user preference vector (average of recent embeddings)
    # We could weight higher rated movies more heavily, but a simple average works well as a baseline.
    user_vector = np.mean(recent_embeddings, axis=0).reshape(1, -1)
    
    # Embed the watchlist
    watchlist_texts = [m.get('title', '') for m in watchlist]
    watchlist_embeddings = model.encode(watchlist_texts)
    
    # Compute cosine similarities
    similarities = cosine_similarity(user_vector, watchlist_embeddings)[0]
    
    # Find the index of the highest similarity
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    
    best_match = watchlist[best_index]
    
    # Format the score as a percentage for the reason string
    match_percentage = int(best_score * 100)
    best_match['reason'] = f"Vibe match! Selected via semantic similarity ({match_percentage}% match) based on your recent activity."
    
    return best_match
