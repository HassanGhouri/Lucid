from backend.app.retrieval.qdrant_store import (
    get_qdrant_client,
    recreate_collection,
)
from backend.app.config import settings

client = get_qdrant_client()
recreate_collection(client, settings.dense_vector_size)

print("Qdrant collection recreated.")
