import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "autonomous_driving_data"
EMBEDDING_DIM = 512

_CACHED_CLIENT = None

def get_qdrant_client(host: str = "localhost", port: int = 6333, storage_path: str = "./qdrant_storage") -> QdrantClient:
    """
    Returns a QdrantClient instance with automatic failover:
    1. Remote Qdrant server (localhost:6333)
    2. Embedded disk storage (./qdrant_storage)
    3. In-memory storage (:memory:) as ultimate fallback if file is locked
    """
    global _CACHED_CLIENT
    if _CACHED_CLIENT is not None:
        try:
            _CACHED_CLIENT.get_collections()
            return _CACHED_CLIENT
        except Exception:
            _CACHED_CLIENT = None

    # 1. Try remote Qdrant server
    try:
        client = QdrantClient(host=host, port=port, timeout=1.5)
        client.get_collections()
        print(f"[QdrantManager] Connected to remote Qdrant at {host}:{port}")
        _CACHED_CLIENT = client
        return client
    except Exception:
        pass

    # 2. Try embedded local disk storage
    os.makedirs(storage_path, exist_ok=True)
    try:
        client = QdrantClient(path=storage_path)
        print(f"[QdrantManager] Initialized local embedded Qdrant at '{storage_path}'")
        _CACHED_CLIENT = client
        return client
    except Exception as lock_err:
        print(f"[QdrantManager] Warning: Embedded storage locked or inaccessible ({lock_err}). Using in-memory client fallback.")

    # 3. In-memory fallback (never fails due to file locks)
    client = QdrantClient(":memory:")
    print("[QdrantManager] Initialized in-memory Qdrant client fallback.")
    _CACHED_CLIENT = client
    return client

def init_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME, recreate: bool = False):
    """
    Ensures the target collection exists with cosine distance.
    """
    try:
        exists = client.collection_exists(collection_name)
        if recreate and exists:
            client.delete_collection(collection_name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            print(f"[QdrantManager] Created collection '{collection_name}' (dim={EMBEDDING_DIM}, distance=COSINE)")
        else:
            print(f"[QdrantManager] Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"[QdrantManager] Collection check/init warning: {e}")
