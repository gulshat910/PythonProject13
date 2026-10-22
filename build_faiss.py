import os
import faiss
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS_PER_ARTICLE = 3      # сколько первых фрагментов статьи векторизуем
BATCH = 256
INDEX_FILE = os.path.join("data", "faiss.index")

model = SentenceTransformer(MODEL, device="cpu")
db = sqlite3.connect("data/metadata.db")

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    last_id = int(faiss.vector_to_array(index.id_map).max()) if index.ntotal else 0
    print("Продолжаю с id", last_id)
else:
    index = faiss.IndexIDMap(faiss.IndexFlatIP(model.get_sentence_embedding_dimension()))
    last_id = 0

total = db.execute("SELECT count(*) FROM metadata WHERE chunk_id < ? AND id > ?",
                   (CHUNKS_PER_ARTICLE, last_id)).fetchone()[0]
print("Осталось векторизовать:", total)

cur = db.execute("SELECT id, title, text FROM metadata WHERE chunk_id < ? AND id > ? ORDER BY id",
                 (CHUNKS_PER_ARTICLE, last_id))
done = 0
while True:
    rows = cur.fetchmany(BATCH)
    if not rows:
        break
    vecs = model.encode([f"{t}\n{x}" for _, t, x in rows],
                        normalize_embeddings=True, batch_size=64)
    index.add_with_ids(np.asarray(vecs, dtype="float32"),
                       np.array([r[0] for r in rows], dtype="int64"))
    done += len(rows)
    if done % (BATCH * 20) == 0:
        faiss.write_index(index, INDEX_FILE)
        print(f"{done}/{total}", flush=True)

faiss.write_index(index, INDEX_FILE)
print("Готово. Векторов:", index.ntotal)
