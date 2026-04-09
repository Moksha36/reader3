import os
import pickle
from functools import lru_cache
from typing import Optional

import psycopg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the processed book folders located?
BOOKS_DIRS = [".", "books"]
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql:///reader3")
db_table_ready = False


class NewWordCreate(BaseModel):
    book_id: str
    chapter_index: int
    page_number: int
    sentence: str
    word: str


def get_db_connection():
    return psycopg.connect(DATABASE_URL)


def ensure_new_words_table() -> bool:
    global db_table_ready

    if db_table_ready:
        return True

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS new_words (
                        id BIGSERIAL PRIMARY KEY,
                        book_id TEXT NOT NULL,
                        chapter_index INTEGER NOT NULL,
                        page_number INTEGER NOT NULL DEFAULT 1,
                        sentence TEXT NOT NULL DEFAULT '',
                        word TEXT NOT NULL,
                        normalized_word TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (book_id, chapter_index, normalized_word)
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE new_words
                    ADD COLUMN IF NOT EXISTS page_number INTEGER NOT NULL DEFAULT 1
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE new_words
                    ADD COLUMN IF NOT EXISTS sentence TEXT NOT NULL DEFAULT ''
                    """
                )
            conn.commit()
        db_table_ready = True
        return True
    except psycopg.Error as exc:
        print(f"Could not initialize PostgreSQL new_words table: {exc}")
        return False


def normalize_word(word: str) -> str:
    return " ".join(word.strip().split()).casefold()


def find_book_dir(folder_name: str) -> Optional[str]:
    """Return the directory containing the processed book folder."""
    safe_folder_name = os.path.basename(folder_name)

    for base_dir in BOOKS_DIRS:
        candidate = os.path.join(base_dir, safe_folder_name)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "book.pkl")):
            return candidate

    return None


@app.on_event("startup")
def startup() -> None:
    ensure_new_words_table()

@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    book_dir = find_book_dir(folder_name)
    if not book_dir:
        return None

    file_path = os.path.join(book_dir, "book.pkl")

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []
    seen = set()

    # Scan all configured directories for folders ending in '_data' that have a book.pkl
    for base_dir in BOOKS_DIRS:
        if not os.path.exists(base_dir):
            continue

        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if item in seen or not item.endswith("_data") or not os.path.isdir(item_path):
                continue

            # Try to load it to get the title
            book = load_book_cached(item)
            if book:
                seen.add(item)
                books.append({
                    "id": item,
                    "title": book.metadata.title,
                    "author": ", ".join(book.metadata.authors),
                    "chapters": len(book.spine)
                })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def read_book(request: Request, book_id: str):
    """Render the full book as one continuous paginated document."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "book_id": book_id,
    })

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def redirect_legacy_chapter_route(request: Request, book_id: str, chapter_index: int):
    """Redirect legacy chapter routes into the single-document reader."""
    destination = request.url_for("read_book", book_id=book_id)
    if request.url.query:
        destination = f"{destination}?{request.url.query}"
    return RedirectResponse(url=str(destination), status_code=307)


@app.get("/api/new-words/{book_id}/{chapter_index}")
async def list_new_words(book_id: str, chapter_index: int):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    safe_book_id = os.path.basename(book_id)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT word, page_number, sentence
                FROM new_words
                WHERE book_id = %s AND chapter_index = %s
                ORDER BY created_at ASC, id ASC
                """,
                (safe_book_id, chapter_index),
            )
            rows = cur.fetchall()

    return {
        "book_id": safe_book_id,
        "chapter_index": chapter_index,
        "words": [
            {"word": row[0], "page_number": row[1], "sentence": row[2]}
            for row in rows
        ],
    }


@app.get("/api/new-words/{book_id}")
async def list_new_words_for_book(book_id: str):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    safe_book_id = os.path.basename(book_id)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chapter_index, word, page_number, sentence
                FROM new_words
                WHERE book_id = %s
                ORDER BY chapter_index ASC, created_at ASC, id ASC
                """,
                (safe_book_id,),
            )
            rows = cur.fetchall()

    return {
        "book_id": safe_book_id,
        "words": [
            {
                "chapter_index": row[0],
                "word": row[1],
                "page_number": row[2],
                "sentence": row[3],
            }
            for row in rows
        ],
    }


@app.post("/api/new-words")
async def create_new_word(payload: NewWordCreate):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    safe_book_id = os.path.basename(payload.book_id)
    normalized_word = normalize_word(payload.word)

    if not normalized_word:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    book = load_book_cached(safe_book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if payload.chapter_index < 0 or payload.chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    if payload.page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be positive")

    sentence = " ".join(payload.sentence.strip().split())
    if not sentence:
        raise HTTPException(status_code=400, detail="Sentence cannot be empty")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO new_words (book_id, chapter_index, page_number, sentence, word, normalized_word)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (book_id, chapter_index, normalized_word)
                DO UPDATE SET
                    word = EXCLUDED.word,
                    page_number = EXCLUDED.page_number,
                    sentence = EXCLUDED.sentence
                RETURNING id, word, page_number, sentence, created_at
                """,
                (
                    safe_book_id,
                    payload.chapter_index,
                    payload.page_number,
                    sentence,
                    payload.word.strip(),
                    normalized_word,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    return {
        "id": row[0],
        "book_id": safe_book_id,
        "chapter_index": payload.chapter_index,
        "word": row[1],
        "page_number": row[2],
        "sentence": row[3],
        "created_at": row[4].isoformat(),
    }

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    book_dir = find_book_dir(safe_book_id)
    if not book_dir:
        raise HTTPException(status_code=404, detail="Book not found")

    img_path = os.path.join(book_dir, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
