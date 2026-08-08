"""向量資料庫封裝（取代 NVIDIA VSS 的 Milvus）。

用 ChromaDB 的 persistent client：存 keyframe 的 CLIP embedding + 時間戳 metadata，
查詢時用 text embedding 做 cosine 相似度檢索。
"""
import chromadb

import config


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(config.DB_DIR))
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        try:
            self.client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, metadatas):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def count(self) -> int:
        return self.collection.count()

    def query(self, embedding, top_k: int):
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        hits = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            hits.append({"meta": meta, "distance": dist, "score": 1.0 - dist})
        return hits

    def get_by_ids(self, ids: list[str]) -> dict[str, dict]:
        """依 id（"<video>::<t_sec>"）直接取回 metadata + embedding，不存在的 id 不會出現在結果裡。

        用來在不重跑影像 forward 的情況下，用既有 embedding 算某個時間點附近影格的相似度分數。
        """
        if not ids:
            return {}
        res = self.collection.get(ids=ids, include=["metadatas", "embeddings"])
        return {
            id_: {"meta": meta, "embedding": emb}
            for id_, meta, emb in zip(res["ids"], res["metadatas"], res["embeddings"])
        }
