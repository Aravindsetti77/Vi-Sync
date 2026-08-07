import os
import redis.asyncio as redis
import json
import numpy as np

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# decode_responses=False is useful for bytes, but we will use True to handle strings/JSON easily
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

async def set_cache(key: str, data: dict, expire_seconds: int = 86400):
    """Stores data in Redis. Lists/arrays are json dumped."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_data = json.dumps(data, default=convert_numpy)
    await redis_client.setex(key, expire_seconds, json_data)

async def get_cache(key: str) -> dict:
    """Retrieves data from Redis and parses JSON."""
    data = await redis_client.get(key)
    if data:
        return json.loads(data)
    return None

async def delete_cache(key: str):
    """Deletes a cache entry by key. Used for invalidation when data changes."""
    await redis_client.delete(key)

async def close_cache():
    await redis_client.close()
