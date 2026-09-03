# Letterboxd Vibe Sync 🎬

**Letterboxd Vibe Sync** is a cinematic curation engine that acts as an intelligent movie recommender. It uses semantic similarity and AI text embeddings to match your (or your friend group's) recent viewing activity with movies from your watchlists.

Stop arguing over what to watch tonight—let the algorithm check the group's "vibe" and pick the perfect movie!

## 🚀 Features

- **Single User Vibe Check**: Analyzes your recent Letterboxd activity (last 10 watches) and compares it semantically to your watchlist to find the movie that best matches your current mood.
- **Group Taste Matching**: Add multiple friends' Letterboxd usernames! The system merges everyone's recent watches to calculate a "group vibe centroid" and recommends a movie found on the group's combined watchlists.
- **AI-Powered Recommendations**: Uses the `all-MiniLM-L6-v2` model via `FastEmbed` (ONNX runtime) to deeply understand the context and themes of movie titles rather than relying on simple genres or tags.
- **Blazing Fast**: Scrapes profiles and RSS feeds concurrently using `asyncio` to reduce loading times even when comparing large groups.
- **Clean UI**: A sleek, minimalistic, fast frontend interface built using vanilla HTML/CSS/JS.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **AI / NLP**: `fastembed` (Lightweight ONNX Embeddings), `scikit-learn` (Cosine Similarity)
- **Scraping**: `BeautifulSoup4`, `httpx`, `feedparser`
- **Frontend**: Vanilla HTML, CSS, and JS

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aravindsetti77/Vi-Sync.git
   cd Vi-Sync
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   *Note: On first run, the FastEmbed model (`all-MiniLM-L6-v2`) will be downloaded locally. This might take a moment depending on your internet connection. You can optionally use the `export.py` script to use a custom local model.*

4. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Open the App:**
   Visit `http://127.0.0.1:8000/` in your browser.

## 💡 How It Works

1. **Scraping Data**: The backend fetches recent watches from a user's Letterboxd RSS feed and their watchlist using HTML parsing (`BeautifulSoup`).
2. **Text Embedding**: Recent movie titles are passed through `FastEmbed` to generate mathematical vector embeddings blazingly fast.
3. **Calculating Vibe**: The vectors of recent watches are averaged to create a user (or group) "preference vector".
4. **Matching**: The watchlists are also embedded into vectors, and the system uses Cosine Similarity to find the closest match to the preference vector.

## 📝 License

This project is open-source and available for personal use. Note that this project is not officially affiliated with or endorsed by Letterboxd. Please use respectfully and do not overload their servers.
