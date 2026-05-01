import os
import pickle
import re
import json
import csv
import io
import html
import shutil
import tempfile
import threading
import uuid
from functools import lru_cache
from typing import Optional
from urllib.parse import quote, unquote

import psycopg
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

try:
    from openai import OpenAI
    from openai import OpenAIError
except ModuleNotFoundError:
    OpenAI = None

    class OpenAIError(Exception):
        pass

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry, process_epub, save_to_pickle

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the processed book folders located?
BOOKS_DIRS = [".", "books"]
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql:///reader3")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
TRANSLATION_MODEL = os.environ.get("READER3_OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
TRANSLATION_TARGET_LANGUAGE = (
    os.environ.get("READER3_TRANSLATION_TARGET_LANGUAGE", "English").strip() or "English"
)
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
db_table_ready = False
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None
upload_jobs: dict[str, dict] = {}
upload_jobs_lock = threading.Lock()

VOCABULARY_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "word": {"type": "string"},
        "part_of_speech": {"type": "string"},
        "definition": {"type": "string"},
        "simple_meaning": {"type": "string"},
        "real_contexts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "scenario": {"type": "string"},
                    "meaning": {"type": "string"},
                },
                "required": ["category", "scenario", "meaning"],
                "additionalProperties": False,
            },
        },
        "example_sentence": {"type": "string"},
        "difference": {
            "type": "object",
            "properties": {
                "main_word": {"type": "string"},
                "similar_word_1": {"type": "string"},
                "similar_word_2": {"type": "string"},
            },
            "required": ["main_word", "similar_word_1", "similar_word_2"],
            "additionalProperties": False,
        },
    },
    "required": [
        "word",
        "part_of_speech",
        "definition",
        "simple_meaning",
        "real_contexts",
        "example_sentence",
        "difference",
    ],
    "additionalProperties": False,
}


class NewWordCreate(BaseModel):
    book_id: str
    chapter_index: int
    page_number: int
    sentence: str
    word: str
    ai_explanation: str = ""


class VocabularyToggle(BaseModel):
    memorized: bool


PAGE_TEXT_TARGET = 1000
PAGE_BLOCK_LIMIT = 8
BLOCK_TEXT_TARGET = 500


def get_db_connection():
    return psycopg.connect(DATABASE_URL)


def format_vocabulary_entry(entry: dict) -> str:
    word = " ".join(str(entry.get("word", "")).split())
    part_of_speech = " ".join(str(entry.get("part_of_speech", "")).split())
    definition = " ".join(str(entry.get("definition", "")).split())
    simple_meaning = " ".join(str(entry.get("simple_meaning", "")).split())
    example_sentence = " ".join(str(entry.get("example_sentence", "")).split())
    difference = entry.get("difference", {}) or {}
    contexts = entry.get("real_contexts", []) or []

    heading_word = word.capitalize() if word else "This word"
    meaning_line = f"{heading_word} means {definition}"
    if part_of_speech:
        meaning_line = f"{meaning_line} ({part_of_speech})."
    elif not meaning_line.endswith("."):
        meaning_line = f"{meaning_line}."

    lines = [
        meaning_line,
        "",
        "Simple meaning:",
        "",
        simple_meaning,
        "",
        "Real contexts:",
        "",
    ]

    for index, context in enumerate(contexts[:4], start=1):
        category = " ".join(str(context.get("category", "")).split())
        scenario = " ".join(str(context.get("scenario", "")).split())
        meaning = " ".join(str(context.get("meaning", "")).split())
        lines.extend([
            f"{index}. {category}",
            "",
            scenario,
            "",
            f"Meaning: {meaning}",
            "",
        ])

    lines.extend([
        "Example sentence:",
        "",
        example_sentence,
        "",
        "Difference:",
        "",
        " ".join(str(difference.get("main_word", "")).split()),
        "",
        " ".join(str(difference.get("similar_word_1", "")).split()),
        "",
        " ".join(str(difference.get("similar_word_2", "")).split()),
    ])

    return "\n".join(line for line in lines if line is not None).strip()


def content_disposition_filename(filename: str) -> str:
    ascii_filename = filename.encode("ascii", "ignore").decode("ascii") or "reader3_vocabulary_export.csv"
    ascii_filename = ascii_filename.replace('"', "")
    utf8_filename = quote(filename)
    return f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{utf8_filename}"


def extract_context_meaning(ai_explanation: str) -> str:
    soup = BeautifulSoup(ai_explanation or "", "html.parser")

    for block in soup.select(".meaning-block"):
        title = block.select_one(".meaning-title")
        if not title or " ".join(title.get_text(" ", strip=True).split()).casefold() != "context meaning":
            continue

        text = block.select_one(".meaning-text")
        return " ".join((text or block).get_text(" ", strip=True).split())

    return ""


def extract_pronunciation(ai_explanation: str) -> str:
    soup = BeautifulSoup(ai_explanation or "", "html.parser")

    for block in soup.select(".meaning-block"):
        title = block.select_one(".meaning-title")
        if title and " ".join(title.get_text(" ", strip=True).split()).casefold() == "pronunciation":
            text = block.select_one(".meaning-text")
            return " ".join((text or block).get_text(" ", strip=True).split())

    for block in soup.select(".meaning-block"):
        if block.select_one(".meaning-title"):
            continue

        text = block.select_one(".meaning-text")
        pronunciation = " ".join((text or block).get_text(" ", strip=True).split())
        if pronunciation.startswith("/") and pronunciation.endswith("/"):
            return pronunciation

    return ""


def vocabulary_export_row(word: str, ai_explanation: str) -> list[str]:
    return [
        word,
        "",
        "",
        extract_pronunciation(ai_explanation),
        "",
        "",
        "",
        extract_context_meaning(ai_explanation),
    ]


def csv_response(rows: list[list[str]], filename: str) -> Response:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)

    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": content_disposition_filename(filename)},
    )


def flattened_toc_entries(entries: list[TOCEntry]) -> list[TOCEntry]:
    flattened = []
    for entry in entries:
        flattened.append(entry)
        flattened.extend(flattened_toc_entries(entry.children))
    return flattened


def chapter_display_number(book: Book, chapter_index: int) -> int:
    if 0 <= chapter_index < len(book.spine):
        chapter_href = book.spine[chapter_index].href
        for entry in flattened_toc_entries(book.toc):
            if entry.file_href != chapter_href:
                continue

            match = re.search(r"\bchapter\s+(\d+)\b", entry.title or "", re.IGNORECASE)
            if match:
                return int(match.group(1))

    return chapter_index + 1


def translate_word_with_openai(word: str, sentence: str) -> str:
    if not openai_client:
        return ""

    try:
        response = openai_client.responses.create(
            model=TRANSLATION_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                    "Return HTML only. "
                    "Do not return markdown. "
                    "Do not use <center>. "
                    "Do not wrap everything in one <p>. "

                    "Output exactly three blocks in this structure: "

                    '<div class="meaning-block">'
                    '<div class="meaning-text">/.../</div>'
                    '</div>'

                    '<div class="meaning-block">'
                    '<div class="meaning-title">Context meaning</div>'
                    '<div class="meaning-text">...</div>'
                    '</div>'

                    '<div class="meaning-block">'
                    '<div class="meaning-title">General meaning</div>'
                    '<ol class="meaning-list">'
                    '<li>...</li>'
                    '<li>...</li>'
                    '</ol>'
                    '</div>'

                    "Rules: "
                    "For the first block, provide only the word pronunciation using IPA symbols, wrapped in slash marks. "
                    "Do not include a title or label for the pronunciation block. "
                    "If the selected text is a phrase, provide a natural pronunciation for the full phrase. "
                    "For the second block, explain the meaning of the word in this specific sentence only. "
                    "Keep it short, clear, and concise. "

                    "For the third block, give 1 to 3 general meanings of the word. "
                    "Each meaning must be one <li>. "

                    "Keep the second and third block titles exactly as "
                    "'Context meaning' and 'General meaning'. "
                    "Output valid HTML fragments only."
                ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Word: {word}\n"
                        f"Sentence: {sentence}\n"
                        f"Explain what '{word}' means in the context of this sentence."
                    ),
                },
            ],
            text={
                "format": {
                    "type": "text",
                }
            },
        )
    except OpenAIError as exc:
        print(f"OpenAI translation failed: {exc}")
        return ""

    explanation = (response.output_text or "").strip()
    return explanation


def populate_ai_explanation(
    book_id: str,
    chapter_index: int,
    page_number: int,
    normalized_word: str,
    word: str,
    sentence: str,
) -> None:
    explanation = translate_word_with_openai(word, sentence)
    if not explanation:
        return

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE new_words
                    SET ai_explanation = %s
                    WHERE book_id = %s
                      AND chapter_index = %s
                      AND page_number = %s
                      AND normalized_word = %s
                      AND ai_explanation = ''
                    """,
                    (explanation, book_id, chapter_index, page_number, normalized_word),
                )
            conn.commit()
    except psycopg.Error as exc:
        print(f"Could not update AI explanation for {word}: {exc}")


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
                        times INTEGER NOT NULL DEFAULT 0,
                        sentence TEXT NOT NULL DEFAULT '',
                        ai_explanation TEXT NOT NULL DEFAULT '',
                        memorized BOOLEAN NOT NULL DEFAULT FALSE,
                        word TEXT NOT NULL,
                        normalized_word TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (book_id, chapter_index, page_number, normalized_word)
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
                    ADD COLUMN IF NOT EXISTS times INTEGER NOT NULL DEFAULT 0
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE new_words
                    ADD COLUMN IF NOT EXISTS sentence TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE new_words
                    ADD COLUMN IF NOT EXISTS ai_explanation TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE new_words
                    ADD COLUMN IF NOT EXISTS memorized BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conname = 'new_words_book_id_chapter_index_normalized_word_key'
                        ) THEN
                            ALTER TABLE new_words
                            DROP CONSTRAINT new_words_book_id_chapter_index_normalized_word_key;
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conname = 'new_words_book_id_chapter_index_page_number_normalized_word_key'
                        ) THEN
                            ALTER TABLE new_words
                            ADD CONSTRAINT new_words_book_id_chapter_index_page_number_normalized_word_key
                            UNIQUE (book_id, chapter_index, page_number, normalized_word);
                        END IF;
                    END
                    $$;
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


def highlight_word_in_sentence(sentence: str, word: str) -> str:
    escaped_sentence = html.escape(sentence or "")
    normalized_word = " ".join((word or "").split()).strip()
    if not escaped_sentence or not normalized_word:
        return escaped_sentence

    pattern = re.compile(re.escape(normalized_word), re.IGNORECASE)
    matches = list(pattern.finditer(sentence or ""))
    if not matches:
        return escaped_sentence

    parts = []
    last_index = 0
    for match in matches:
        parts.append(html.escape(sentence[last_index:match.start()]))
        parts.append(f"<mark>{html.escape(sentence[match.start():match.end()])}</mark>")
        last_index = match.end()
    parts.append(html.escape(sentence[last_index:]))
    return "".join(parts)


def slugify_book_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "book"


def unique_book_output_dir(base_slug: str, root_dir: str = "books") -> tuple[str, str]:
    os.makedirs(root_dir, exist_ok=True)
    candidate_slug = base_slug
    suffix = 2

    while True:
        output_dir = os.path.join(root_dir, f"{candidate_slug}_data")
        if not os.path.exists(output_dir):
            return candidate_slug, output_dir
        candidate_slug = f"{base_slug}_{suffix}"
        suffix += 1


def clear_book_caches() -> None:
    load_book_cached.cache_clear()
    paginate_book_cached.cache_clear()


def set_upload_job(job_id: str, **updates) -> None:
    with upload_jobs_lock:
        job = upload_jobs.get(job_id, {}).copy()
        job.update(updates)
        upload_jobs[job_id] = job


def get_upload_job(job_id: str) -> Optional[dict]:
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        return job.copy() if job else None


def stage_uploaded_book(upload: UploadFile) -> tuple[str, str]:
    filename = os.path.basename(upload.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="Please choose a file to upload.")

    extension = os.path.splitext(filename)[1].lower()
    if extension != ".epub":
        raise HTTPException(status_code=400, detail="Only EPUB uploads are supported.")

    total_size = 0
    temp_dir = tempfile.mkdtemp(prefix="reader3_upload_")
    temp_book_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_book_path, "wb") as temp_file:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
                    )
                temp_file.write(chunk)
        return temp_dir, temp_book_path
    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        try:
            upload.file.close()
        except Exception:
            pass


def process_uploaded_book_job(job_id: str, temp_dir: str, temp_book_path: str) -> None:
    filename = os.path.basename(temp_book_path)
    try:
        set_upload_job(job_id, status="processing", message=f"Processing {filename}...")
        base_slug = slugify_book_name(os.path.splitext(filename)[0])
        _, output_dir = unique_book_output_dir(base_slug)
        book_obj = process_epub(temp_book_path, output_dir)
        save_to_pickle(book_obj, output_dir)
        shutil.copy2(temp_book_path, os.path.join(output_dir, filename))
        clear_book_caches()

        set_upload_job(
            job_id,
            status="completed",
            message=f'Added "{book_obj.metadata.title}" to your library.',
            book={
                "id": os.path.basename(output_dir),
                "title": book_obj.metadata.title,
                "author": ", ".join(book_obj.metadata.authors) or "Unknown author",
                "chapters": len(book_obj.spine),
            },
        )
    except Exception as exc:
        set_upload_job(job_id, status="error", message=f"Could not process book: {exc}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def find_book_dir(folder_name: str) -> Optional[str]:
    """Return the directory containing the processed book folder."""
    safe_folder_name = os.path.basename(folder_name)

    for base_dir in BOOKS_DIRS:
        candidate = os.path.join(base_dir, safe_folder_name)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "book.pkl")):
            return candidate

    return None


def list_books_index() -> dict[str, dict[str, str]]:
    books = {}
    seen = set()

    for base_dir in BOOKS_DIRS:
        if not os.path.exists(base_dir):
            continue

        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if item in seen or not item.endswith("_data") or not os.path.isdir(item_path):
                continue

            book = load_book_cached(item)
            if book:
                seen.add(item)
                books[item] = {
                    "title": book.metadata.title,
                    "author": ", ".join(book.metadata.authors) or "Unknown author",
                }

    return books


LEAF_BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "figure", "table", "ul", "ol", "img", "hr",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body"}
MEDIA_TAGS = {"img", "svg", "figure", "table", "pre", "hr"}
TEXT_SPLIT_TAGS = {"p", "blockquote", "li"}


def block_weight(tag: Tag) -> int:
    text_len = len(" ".join(tag.get_text(" ", strip=True).split()))
    media_count = len(tag.find_all(list(MEDIA_TAGS)))
    if tag.name in MEDIA_TAGS:
        media_count += 1
    return max(text_len, media_count * 120, 1)


def split_long_text_tag(tag: Tag) -> list[tuple[str, int]]:
    text = " ".join(tag.get_text(" ", strip=True).split())
    if len(text) <= BLOCK_TEXT_TARGET:
        return [(str(tag), max(len(text), 1))]

    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()
    ]
    if len(sentences) <= 1:
        words = text.split()
        if len(words) <= 1:
            return [(str(tag), max(len(text), 1))]
        sentences = []
        current_words: list[str] = []
        for word in words:
            current_words.append(word)
            if len(" ".join(current_words)) >= BLOCK_TEXT_TARGET:
                sentences.append(" ".join(current_words))
                current_words = []
        if current_words:
            sentences.append(" ".join(current_words))

    chunks: list[str] = []
    current_chunk = ""
    for sentence in sentences:
        candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        if current_chunk and len(candidate) > BLOCK_TEXT_TARGET:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = candidate
    if current_chunk:
        chunks.append(current_chunk)

    attrs = "".join(f' {key}="{value}"' for key, value in tag.attrs.items())
    return [
        (f"<{tag.name}{attrs}>{chunk}</{tag.name}>", len(chunk))
        for chunk in chunks
    ]


def collect_blocks(node) -> list[tuple[str, int]]:
    if isinstance(node, NavigableString):
        text = " ".join(str(node).split())
        return [(f"<p>{text}</p>", len(text))] if text else []

    if not isinstance(node, Tag):
        return []

    if node.name in LEAF_BLOCK_TAGS:
        if node.name in TEXT_SPLIT_TAGS and not node.find(list(MEDIA_TAGS)):
            return split_long_text_tag(node)
        return [(str(node), block_weight(node))]

    direct_tag_children = [child for child in node.children if isinstance(child, Tag)]
    direct_block_children = [
        child for child in direct_tag_children
        if child.name in LEAF_BLOCK_TAGS or child.name in CONTAINER_TAGS
    ]

    if node.name in CONTAINER_TAGS and len(direct_block_children) > 1:
        blocks: list[tuple[str, int]] = []
        for child in node.children:
            blocks.extend(collect_blocks(child))
        return blocks

    block_html = str(node)
    weight = block_weight(node)
    if weight > 1 or block_html.strip():
        return [(block_html, weight)]

    return []


def top_level_blocks(html: str) -> list[tuple[str, int]]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.body or soup
    blocks: list[tuple[str, int]] = []

    for child in root.contents:
        blocks.extend(collect_blocks(child))

    if not blocks:
        fallback_html = html.strip()
        if fallback_html:
            fallback_text = " ".join(soup.get_text(" ", strip=True).split())
            blocks.append((fallback_html, max(len(fallback_text), 120)))

    return blocks


def paginate_chapter(chapter: ChapterContent) -> list[dict]:
    blocks = top_level_blocks(chapter.content)
    if not blocks:
        return [{
            "chapter_index": chapter.order,
            "chapter_href": chapter.href,
            "html": f'<section class="book-section" data-chapter-index="{chapter.order}" data-chapter-href="{chapter.href}"></section>',
        }]

    pages: list[dict] = []
    current_blocks: list[str] = []
    current_text = 0

    for block_html, text_len in blocks:
        would_overflow = (
            current_blocks and (
                current_text + text_len > PAGE_TEXT_TARGET or
                len(current_blocks) >= PAGE_BLOCK_LIMIT
            )
        )

        if would_overflow:
            pages.append({
                "chapter_index": chapter.order,
                "chapter_href": chapter.href,
                "html": (
                    f'<section class="book-section" '
                    f'data-chapter-index="{chapter.order}" '
                    f'data-chapter-href="{chapter.href}">'
                    + "".join(current_blocks) +
                    "</section>"
                ),
            })
            current_blocks = []
            current_text = 0

        current_blocks.append(block_html)
        current_text += text_len

    if current_blocks:
        pages.append({
            "chapter_index": chapter.order,
            "chapter_href": chapter.href,
            "html": (
                f'<section class="book-section" '
                f'data-chapter-index="{chapter.order}" '
                f'data-chapter-href="{chapter.href}">'
                + "".join(current_blocks) +
                "</section>"
            ),
        })

    return pages


def page_html_for_book(page_html: str, book_id: str, first_page_for_href: dict[str, int]) -> str:
    soup = BeautifulSoup(page_html, "html.parser")
    book_path = quote(book_id, safe="")

    for tag in soup.find_all(src=True):
        src = tag.get("src", "")
        if src.startswith(("http://", "https://", "/", "data:", "#")):
            continue
        if src.startswith("images/"):
            tag["src"] = f"/read/{book_path}/{src}"

    for tag in soup.find_all(href=True):
        href = tag.get("href", "")
        if not href or href.startswith(("http://", "https://", "/", "mailto:", "tel:", "javascript:")):
            continue

        file_href, _, anchor = href.partition("#")
        if not file_href:
            continue

        target_page = first_page_for_href.get(file_href) or first_page_for_href.get(unquote(file_href))
        if target_page:
            rewritten_href = f"/read/{book_path}?page={target_page}"
            if anchor:
                rewritten_href = f"{rewritten_href}#{quote(anchor, safe='')}"
            tag["href"] = rewritten_href

    return str(soup)


@lru_cache(maxsize=10)
def paginate_book_cached(book_id: str) -> Optional[dict]:
    book = load_book_cached(book_id)
    if not book:
        return None

    pages: list[dict] = []
    first_page_for_href: dict[str, int] = {}

    for chapter in book.spine:
        chapter_pages = paginate_chapter(chapter)
        first_page = len(pages) + 1
        href_keys = {
            chapter.href,
            unquote(chapter.href),
            quote(unquote(chapter.href), safe="/."),
        }
        for href_key in href_keys:
            if href_key and href_key not in first_page_for_href:
                first_page_for_href[href_key] = first_page

        for page in chapter_pages:
            pages.append(page)

    return {
        "pages": pages,
        "first_page_for_href": first_page_for_href,
    }


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
    for book_id, info in list_books_index().items():
        book = load_book_cached(book_id)
        if not book:
            continue
        books.append({
            "id": book_id,
            "title": info["title"],
            "author": info["author"],
            "chapters": len(book.spine)
        })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})


@app.post("/api/books/upload")
async def upload_book(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_dir, temp_book_path = stage_uploaded_book(file)
    job_id = uuid.uuid4().hex
    set_upload_job(
        job_id,
        status="queued",
        message=f"Upload complete. Processing {os.path.basename(temp_book_path)}...",
    )
    background_tasks.add_task(process_uploaded_book_job, job_id, temp_dir, temp_book_path)
    return {"ok": True, "job_id": job_id}


@app.get("/api/books/upload/{job_id}")
async def upload_book_status(job_id: str):
    job = get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job not found.")
    return {"ok": True, "job": job}


@app.post("/books/delete")
async def delete_book(book_id: str = Form(...)):
    safe_book_id = os.path.basename(book_id)
    book_dir = find_book_dir(safe_book_id)
    if not book_dir:
        raise HTTPException(status_code=404, detail="Book not found")

    shutil.rmtree(book_dir, ignore_errors=True)

    if ensure_new_words_table():
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM new_words
                        WHERE book_id = %s
                        """,
                        (safe_book_id,),
                    )
                conn.commit()
        except psycopg.Error as exc:
            print(f"Could not delete saved words for {safe_book_id}: {exc}")

    clear_book_caches()
    return RedirectResponse(url="/", status_code=303)

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def read_book(request: Request, book_id: str):
    safe_book_id = os.path.basename(book_id)
    book = load_book_cached(safe_book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    paginated = paginate_book_cached(safe_book_id)
    if not paginated or not paginated["pages"]:
        raise HTTPException(status_code=404, detail="Book pages not found")

    page = request.query_params.get("page", "1")
    try:
        current_page = int(page)
    except ValueError:
        current_page = 1

    current_page = max(1, min(current_page, len(paginated["pages"])))
    page_data = paginated["pages"][current_page - 1]
    prev_page = current_page - 1 if current_page > 1 else None
    next_page = current_page + 1 if current_page < len(paginated["pages"]) else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "book_id": safe_book_id,
        "current_page": current_page,
        "total_pages": len(paginated["pages"]),
        "current_page_html": page_html_for_book(
            page_data["html"],
            safe_book_id,
            paginated["first_page_for_href"],
        ),
        "current_chapter_index": page_data["chapter_index"],
        "prev_page": prev_page,
        "next_page": next_page,
        "toc_page_map": paginated["first_page_for_href"],
    })

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def redirect_legacy_chapter_route(request: Request, book_id: str, chapter_index: int):
    safe_book_id = os.path.basename(book_id)
    book = load_book_cached(safe_book_id)
    paginated = paginate_book_cached(safe_book_id)
    if not book or not paginated:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    destination = request.url_for("read_book", book_id=safe_book_id)
    first_page = paginated["first_page_for_href"].get(book.spine[chapter_index].href, 1)
    destination = f"{destination}?page={first_page}"
    return RedirectResponse(url=str(destination), status_code=307)


@app.get("/api/new-words/{book_id}/{chapter_index}")
async def list_new_words(book_id: str, chapter_index: int, page_number: Optional[int] = None):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    safe_book_id = os.path.basename(book_id)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if page_number is not None:
                cur.execute(
                    """
                    SELECT word, page_number, times, sentence, ai_explanation, memorized
                    FROM new_words
                    WHERE book_id = %s AND chapter_index = %s AND page_number = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (safe_book_id, chapter_index, page_number),
                )
            else:
                cur.execute(
                    """
                    SELECT word, page_number, times, sentence, ai_explanation, memorized
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
            {
                "word": row[0],
                "page_number": row[1],
                "times": row[2],
                "sentence": row[3],
                "ai_explanation": row[4],
                "memorized": row[5],
            }
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
                SELECT chapter_index, word, page_number, times, sentence, ai_explanation, memorized
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
                "times": row[3],
                "sentence": row[4],
                "ai_explanation": row[5],
                "memorized": row[6],
            }
            for row in rows
        ],
    }


@app.post("/api/new-words")
async def create_new_word(payload: NewWordCreate, background_tasks: BackgroundTasks):
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
    ai_explanation = " ".join(payload.ai_explanation.strip().split())
    should_generate_ai_explanation = not ai_explanation

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO new_words (book_id, chapter_index, page_number, times, sentence, ai_explanation, word, normalized_word)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (book_id, chapter_index, page_number, normalized_word)
                DO UPDATE SET
                    word = EXCLUDED.word,
                    page_number = EXCLUDED.page_number,
                    sentence = EXCLUDED.sentence,
                    ai_explanation = CASE
                        WHEN EXCLUDED.ai_explanation <> '' THEN EXCLUDED.ai_explanation
                        ELSE new_words.ai_explanation
                    END
                RETURNING id, word, page_number, times, sentence, ai_explanation, memorized, created_at
                """,
                (
                    safe_book_id,
                    payload.chapter_index,
                    payload.page_number,
                    0,
                    sentence,
                    ai_explanation,
                    payload.word.strip(),
                    normalized_word,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if should_generate_ai_explanation and not row[5]:
        background_tasks.add_task(
            populate_ai_explanation,
            safe_book_id,
            payload.chapter_index,
            payload.page_number,
            normalized_word,
            payload.word.strip(),
            sentence,
        )

    return {
        "id": row[0],
        "book_id": safe_book_id,
        "chapter_index": payload.chapter_index,
        "word": row[1],
        "page_number": row[2],
        "times": row[3],
        "sentence": row[4],
        "ai_explanation": row[5],
        "ai_explanation_pending": should_generate_ai_explanation and not row[5],
        "memorized": row[6],
        "created_at": row[7].isoformat(),
    }


@app.get("/vocabulary", response_class=HTMLResponse)
async def vocabulary_view(request: Request, book_id: Optional[str] = None):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    books_index = list_books_index()
    safe_book_id = os.path.basename(book_id) if book_id else None

    if safe_book_id and safe_book_id not in books_index:
        raise HTTPException(status_code=404, detail="Book not found")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if safe_book_id:
                cur.execute(
                    """
                    SELECT id, book_id, chapter_index, page_number, times, word, sentence, ai_explanation, memorized, created_at
                    FROM new_words
                    WHERE book_id = %s
                    ORDER BY memorized ASC, created_at DESC, id DESC
                    """,
                    (safe_book_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, book_id, chapter_index, page_number, times, word, sentence, ai_explanation, memorized, created_at
                    FROM new_words
                    ORDER BY memorized ASC, created_at DESC, id DESC
                    """
                )
            rows = cur.fetchall()

    words = [row[5] for row in rows]
    chapter_counts: list[dict[str, object]] = []
    chapter_totals: dict[tuple[str, int], int] = {}

    for row in rows:
        key = (row[1], row[2])
        chapter_totals[key] = chapter_totals.get(key, 0) + 1

    for (row_book_id, chapter_index), count in sorted(
        chapter_totals.items(),
        key=lambda item: (
            books_index.get(item[0][0], {}).get("title", item[0][0]).casefold(),
            item[0][1],
        ),
    ):
        chapter_counts.append({
            "label": f"{books_index.get(row_book_id, {}).get('title', row_book_id)} - Chapter {chapter_index}",
            "count": count,
        })

    sheet_columns = 20
    sheet_rows = 20
    sheet_size = sheet_columns * sheet_rows
    word_sheets = []
    for index in range(0, len(words), sheet_size):
        sheet = words[index:index + sheet_size]
        if len(sheet) < sheet_size:
            sheet = sheet + [""] * (sheet_size - len(sheet))
        grid_rows: list[list[str]] = []
        for row_index in range(sheet_rows):
            grid_rows.append([
                sheet[(column_index * sheet_rows) + row_index]
                for column_index in range(sheet_columns)
            ])
        word_sheets.append(grid_rows)

    return templates.TemplateResponse("vocabulary.html", {
        "request": request,
        "words": words,
        "word_sheets": word_sheets,
        "chapter_counts": chapter_counts,
        "book_id": safe_book_id,
        "book_title": books_index.get(safe_book_id, {}).get("title") if safe_book_id else None,
        "total_count": len(words),
    })


@app.get("/vocabulary/flashcards", response_class=HTMLResponse)
async def vocabulary_flashcards_view(request: Request, book_id: Optional[str] = None):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    books_index = list_books_index()
    safe_book_id = os.path.basename(book_id) if book_id else None

    if safe_book_id and safe_book_id not in books_index:
        raise HTTPException(status_code=404, detail="Book not found")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if safe_book_id:
                cur.execute(
                    """
                    SELECT id, book_id, chapter_index, page_number, word, sentence, ai_explanation, memorized, created_at
                    FROM new_words
                    WHERE book_id = %s
                    ORDER BY memorized ASC, created_at DESC, id DESC
                    """,
                    (safe_book_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, book_id, chapter_index, page_number, word, sentence, ai_explanation, memorized, created_at
                    FROM new_words
                    ORDER BY memorized ASC, created_at DESC, id DESC
                    """
                )
            rows = cur.fetchall()

    books_by_id = {row[1]: load_book_cached(row[1]) for row in rows}
    flashcards = []
    for row in rows:
        row_book = books_by_id.get(row[1])
        flashcards.append({
            "id": row[0],
            "book_id": row[1],
            "chapter_index": row[2],
            "chapter_number": chapter_display_number(row_book, row[2]) if row_book else row[2] + 1,
            "page_number": row[3],
            "word": row[4],
            "sentence": row[5],
            "highlighted_sentence": highlight_word_in_sentence(row[5], row[4]),
            "ai_explanation": row[6],
            "memorized": row[7],
            "created_at": row[8],
            "book_title": books_index.get(row[1], {}).get("title", row[1]),
        })

    return templates.TemplateResponse("flashcards.html", {
        "request": request,
        "flashcards": flashcards,
        "book_id": safe_book_id,
        "book_title": books_index.get(safe_book_id, {}).get("title") if safe_book_id else None,
        "memorized_count": sum(1 for row in flashcards if row["memorized"]),
        "total_count": len(flashcards),
    })


@app.get("/vocabulary/export", response_class=HTMLResponse)
async def export_vocabulary_view(request: Request, book_id: Optional[str] = None):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    books_index = list_books_index()
    safe_book_id = os.path.basename(book_id) if book_id else None

    if safe_book_id and safe_book_id not in books_index:
        raise HTTPException(status_code=404, detail="Book not found")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if safe_book_id:
                cur.execute(
                    """
                    SELECT book_id, chapter_index, COUNT(*)
                    FROM new_words
                    WHERE book_id = %s
                    GROUP BY book_id, chapter_index
                    ORDER BY chapter_index ASC
                    """,
                    (safe_book_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT book_id, chapter_index, COUNT(*)
                    FROM new_words
                    GROUP BY book_id, chapter_index
                    ORDER BY book_id ASC, chapter_index ASC
                    """
                )
            rows = cur.fetchall()

    chapters = []
    for row_book_id, chapter_index, count in rows:
        book = load_book_cached(row_book_id)
        display_number = chapter_index + 1
        if book:
            display_number = chapter_display_number(book, chapter_index)

        chapters.append({
            "book_id": row_book_id,
            "book_title": books_index.get(row_book_id, {}).get("title", row_book_id),
            "chapter_index": chapter_index,
            "chapter_number": display_number,
            "count": count,
        })

    return templates.TemplateResponse("export.html", {
        "request": request,
        "book_id": safe_book_id,
        "book_title": books_index.get(safe_book_id, {}).get("title") if safe_book_id else None,
        "chapters": chapters,
        "total_count": sum(chapter["count"] for chapter in chapters),
    })


@app.get("/vocabulary/export/download")
async def export_vocabulary_download(book_id: Optional[str] = None):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    safe_book_id = os.path.basename(book_id) if book_id else None

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if safe_book_id:
                cur.execute(
                    """
                    SELECT word, ai_explanation
                    FROM new_words
                    WHERE book_id = %s
                    ORDER BY chapter_index ASC, created_at ASC, id ASC
                    """,
                    (safe_book_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT word, ai_explanation
                    FROM new_words
                    ORDER BY book_id ASC, chapter_index ASC, created_at ASC, id ASC
                    """
                )
            rows = cur.fetchall()

    csv_rows = [["word", "", "", "pronunciation", "", "", "", "context meaning"]]
    for word, ai_explanation in rows:
        csv_rows.append(vocabulary_export_row(word, ai_explanation))

    filename = f"{safe_book_id}_vocabulary_export.csv" if safe_book_id else "reader3_vocabulary_export.csv"
    return csv_response(csv_rows, filename)


@app.get("/vocabulary/export/chapter")
async def export_vocabulary_chapter(book_id: str, chapter_index: int):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    safe_book_id = os.path.basename(book_id)
    book = load_book_cached(safe_book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT word, ai_explanation
                FROM new_words
                WHERE book_id = %s AND chapter_index = %s
                ORDER BY created_at ASC, id ASC
                """,
                (safe_book_id, chapter_index),
            )
            rows = cur.fetchall()

    csv_rows = [
        [book.metadata.title],
        [f"Chapter {chapter_display_number(book, chapter_index)}"],
    ]
    for word, ai_explanation in rows:
        csv_rows.append(vocabulary_export_row(word, ai_explanation))

    filename = f"{safe_book_id}_chapter_{chapter_display_number(book, chapter_index)}_vocabulary_export.csv"
    return csv_response(csv_rows, filename)


@app.post("/vocabulary/toggle")
async def toggle_vocabulary_word(
    word_id: int = Form(...),
    memorized: bool = Form(...),
    return_to: str = Form("/vocabulary"),
):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE new_words
                SET memorized = %s
                WHERE id = %s
                """,
                (memorized, word_id),
            )
        conn.commit()

    return RedirectResponse(url=return_to, status_code=303)


@app.post("/vocabulary/delete")
async def delete_vocabulary_word(
    word_id: int = Form(...),
    return_to: str = Form("/vocabulary"),
):
    if not ensure_new_words_table():
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM new_words
                WHERE id = %s
                """,
                (word_id,),
            )
        conn.commit()

    return RedirectResponse(url=return_to, status_code=303)


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
