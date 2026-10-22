import json, os, sqlite3

FILES = [r"C:\Users\Gulshat\Desktop\habr_clean.jsonl",
         r"C:\Users\Gulshat\Desktop\habr_clean2.jsonl"]
os.makedirs("data", exist_ok=True)


def split(text, size=1000):
    parts = []
    for p in text.split("\n\n"):
        while len(p) > size:
            parts.append(p[:size]); p = p[size:]
        if p.strip():
            parts.append(p)
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > size:
            out.append(buf.strip()); buf = ""
        buf += p + "\n\n"
    if buf.strip():
        out.append(buf.strip())
    return out


db = sqlite3.connect("data/metadata.db")
db.executescript("""
DROP TABLE IF EXISTS fts;
DROP TABLE IF EXISTS metadata;
CREATE TABLE metadata(
    id INTEGER PRIMARY KEY, article_id TEXT, title TEXT, url TEXT,
    published TEXT, hubs TEXT, tags TEXT, chunk_id INTEGER, text TEXT);
""")

seen, n = set(), 0
for path in FILES:
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a["id"] in seen or a["lang"] != "ru" or len(a["text"]) < 300:
                continue
            seen.add(a["id"])
            rows = [(a["id"], a["title"], a["url"], a["published"] or "",
                     json.dumps(a["hubs"], ensure_ascii=False),
                     json.dumps(a["tags"], ensure_ascii=False), i, ch)
                    for i, ch in enumerate(split(a["text"]))]
            db.executemany(
                "INSERT INTO metadata(article_id,title,url,published,hubs,tags,chunk_id,text)"
                " VALUES (?,?,?,?,?,?,?,?)", rows)
            n += 1
            if n % 5000 == 0:
                db.commit(); print("статей:", n, flush=True)
db.commit()

print("Строю полнотекстовый индекс...")
db.executescript("""
CREATE VIRTUAL TABLE fts USING fts5(
    title, text, content='metadata', content_rowid='id', tokenize='unicode61');
INSERT INTO fts(fts) VALUES('rebuild');
""")
db.commit()
print("Готово. Статей:", n, "Фрагментов:",
      db.execute("SELECT count(*) FROM metadata").fetchone()[0])
