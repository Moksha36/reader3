"""PDF ingestion for Reader3.

The original PDF remains the visual source of truth.  This module extracts only
the metadata, outline, and page text needed by the library, vocabulary context,
and navigation layers.
"""

import os
import shutil
from datetime import datetime

import pymupdf

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry


OCR_TEXT_THRESHOLD = 20
OCR_DPI = int(os.environ.get("READER3_OCR_DPI", "180"))
OCR_LANGUAGE = os.environ.get("READER3_OCR_LANGUAGE", "eng").strip() or "eng"


def _add_searchable_ocr_text(page: pymupdf.Page) -> str:
    """OCR one page and add invisible, positioned text to the PDF itself."""
    text_page = page.get_textpage_ocr(
        language=OCR_LANGUAGE,
        dpi=OCR_DPI,
        full=True,
    )
    words = page.get_text("words", textpage=text_page, sort=True)
    inserted_words: list[str] = []

    for x0, y0, x1, y1, word, *_ in words:
        cleaned_word = " ".join(str(word).split())
        if not cleaned_word:
            continue
        height = max(1.0, y1 - y0)
        font_size = max(3.0, height * 0.78)
        baseline = y1 - (height * 0.12)
        page.insert_text(
            (x0, baseline),
            f"{cleaned_word} ",
            fontsize=font_size,
            fontname="helv",
            render_mode=3,
            overlay=True,
        )
        inserted_words.append(cleaned_word)

    return " ".join(inserted_words)


def _metadata(document: pymupdf.Document, source_path: str) -> BookMetadata:
    values = document.metadata or {}
    title = (values.get("title") or "").strip() or os.path.splitext(os.path.basename(source_path))[0]
    author = (values.get("author") or "").strip()
    return BookMetadata(
        title=title,
        language="",
        authors=[author] if author else [],
        description=(values.get("subject") or "").strip() or None,
        publisher=None,
        date=(values.get("creationDate") or "").strip() or None,
        identifiers=[],
        subjects=[],
    )


def _outline(document: pymupdf.Document) -> list[TOCEntry]:
    raw_entries = document.get_toc(simple=True) or []
    roots: list[TOCEntry] = []
    stack: list[tuple[int, TOCEntry]] = []

    for level, title, page_number, *_ in raw_entries:
        if page_number < 1 or page_number > document.page_count:
            continue
        entry = TOCEntry(
            title=(title or f"Page {page_number}").strip(),
            href=f"page-{page_number}",
            file_href=f"page-{page_number}",
            anchor="",
        )
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1].children.append(entry)
        else:
            roots.append(entry)
        stack.append((level, entry))

    if roots:
        return roots

    return [
        TOCEntry(
            title=f"Page {page_number}",
            href=f"page-{page_number}",
            file_href=f"page-{page_number}",
            anchor="",
        )
        for page_number in range(1, document.page_count + 1)
    ]


def process_pdf(pdf_path: str, output_dir: str) -> Book:
    source_filename = os.path.basename(pdf_path)
    probe = pymupdf.open(pdf_path)
    try:
        if probe.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")
        if probe.page_count < 1:
            raise ValueError("The PDF does not contain any pages.")
    finally:
        probe.close()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    searchable_pdf_path = os.path.join(output_dir, source_filename)
    shutil.copy2(pdf_path, searchable_pdf_path)

    document = pymupdf.open(searchable_pdf_path)
    ocr_page_count = 0
    try:
        pages: list[ChapterContent] = []
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            text = " ".join(page.get_text("text", sort=True).split())
            if len(text) <= OCR_TEXT_THRESHOLD:
                try:
                    ocr_text = _add_searchable_ocr_text(page)
                except RuntimeError as exc:
                    print(f"OCR failed for page {page_number}: {exc}")
                    ocr_text = ""
                if ocr_text:
                    text = ocr_text
                    ocr_page_count += 1
            pages.append(
                ChapterContent(
                    id=f"page-{page_number}",
                    href=f"page-{page_number}",
                    title=f"Page {page_number}",
                    content="",
                    text=text,
                    order=page_index,
                )
            )

        book = Book(
            metadata=_metadata(document, pdf_path),
            spine=pages,
            toc=_outline(document),
            images={},
            source_file=source_filename,
            processed_at=datetime.now().isoformat(),
            version="4.0",
            format="pdf",
            page_count=document.page_count,
            ocr_page_count=ocr_page_count,
        )
        if ocr_page_count:
            document.saveIncr()
        return book
    finally:
        document.close()
