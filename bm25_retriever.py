import re
from src.db import connect, row_to_doc, COLS

STOP = {"что", "как", "какой", "какая", "какие", "каком", "какое", "какую", "это",
        "для", "при", "или", "его", "она", "они", "был", "была", "были", "кто",
        "где", "когда", "сколько", "через", "про", "над", "под", "все", "так"}


def make_match(query):
    """Запрос -> FTS5. Префикс из 6 букв даёт грубую морфологию: компан* = компания/компании."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3 and w not in STOP]
    terms = list(dict.fromkeys(w[:6] for w in words))
    return " OR ".join(f'"{t}"*' for t in terms)


def search_bm25(query, top_k=20):
    match = make_match(query)
    if not match:
        return []
    db = connect()
    rows = db.execute(
        f"SELECT {', '.join('m.' + c.strip() for c in COLS.split(','))}, bm25(fts, 3.0, 1.0) AS s "
        "FROM fts JOIN metadata m ON m.id = fts.rowid "
        "WHERE fts MATCH ? ORDER BY s LIMIT ?", (match, top_k)).fetchall()
    db.close()
    return [row_to_doc(r, -r[9], "bm25") for r in rows]   # в SQLite меньше = лучше

if __name__ == "__main__":
    q = input("Введите вопрос: ")
    for i, d in enumerate(search_bm25(q, 5), 1):
        print(f"{i}. {d.metadata['title']} (score {d.metadata['score']:.2f})")
        print(d.page_content[:300], "\n")
