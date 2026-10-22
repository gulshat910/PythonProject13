import numpy as np
from src.db import connect, row_to_doc, COLS
from src.vectorstore import index, embeddings


def search_documents(query, top_k=20):
    q = np.array([embeddings.embed_query(query)], dtype="float32")
    scores, ids = index.search(q, top_k)
    ids = [int(i) for i in ids[0] if i != -1]
    if not ids:
        return []
    db = connect()
    rows = {r[0]: r for r in db.execute(
        f"SELECT {COLS} FROM metadata WHERE id IN ({','.join('?' * len(ids))})", ids)}
    db.close()
    return [row_to_doc(rows[i], s, "vector")
            for i, s in zip(ids, scores[0]) if i in rows]
