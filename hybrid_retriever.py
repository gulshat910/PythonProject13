import numpy as np
from langchain_core.retrievers import BaseRetriever
from src.retriever import search_documents
from src.bm25_retriever import search_bm25
from src.vectorstore import embeddings


def reciprocal_rank_fusion(*doc_lists, k=60):
    """RRF с равными весами."""
    scores, docs = {}, {}
    for lst in doc_lists:
        for rank, d in enumerate(lst, 1):
            key = d.metadata["id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs.setdefault(key, d)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def rerank(query, docs, top_k):
    """Финальный порядок по косинусной близости к вопросу."""
    if not docs:
        return []
    q = np.array(embeddings.embed_query(query))
    v = np.array(embeddings.embed_documents(
        [d.metadata["title"] + "\n" + d.page_content for d in docs]))
    sims = v @ q
    for d, s in zip(docs, sims):
        d.metadata["score"] = float(s)
    return [docs[i] for i in np.argsort(-sims)[:top_k]]


class HybridRetriever(BaseRetriever):
    top_k: int = 8

    def _get_relevant_documents(self, query, *, run_manager=None):
        vec = search_documents(query, 20)
        bm = search_bm25(query, 20)
        fused = reciprocal_rank_fusion(vec, bm)
        results = fused[:self.top_k]
        for rank, d in enumerate(results, 1):
            d.metadata["score"] = 1 / (60 + rank)   # RRF-оценка для показа
        return results


retriever = HybridRetriever(top_k=8)
