import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import pymupdf
from fastapi.testclient import TestClient

import server
from pdf_processor import process_pdf
from reader3 import save_to_pickle


class PdfSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_path = os.path.join(self.temp_dir.name, "sample.pdf")
        document = pymupdf.open()
        for text in ("First page has selectable vocabulary.", "Second page keeps context."):
            page = document.new_page()
            page.insert_text((72, 100), text, fontsize=14)
        document.set_toc([[1, "Start", 1], [1, "Continue", 2]])
        document.set_metadata({"title": "Sample PDF", "author": "Reader3 Tests"})
        document.save(self.source_path)
        document.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_pdf_extracts_pages_metadata_text_and_outline(self):
        output_dir = os.path.join(self.temp_dir.name, "sample_data")
        book = process_pdf(self.source_path, output_dir)

        self.assertEqual(book.format, "pdf")
        self.assertEqual(book.metadata.title, "Sample PDF")
        self.assertEqual(book.metadata.authors, ["Reader3 Tests"])
        self.assertEqual(book.page_count, 2)
        self.assertEqual(len(book.spine), 2)
        self.assertIn("selectable vocabulary", book.spine[0].text)
        self.assertEqual([entry.title for entry in book.toc], ["Start", "Continue"])

    def test_pdf_reader_and_source_routes(self):
        output_dir = os.path.join(self.temp_dir.name, "sample_data")
        book = process_pdf(self.source_path, output_dir)
        shutil.copy2(self.source_path, os.path.join(output_dir, book.source_file))
        save_to_pickle(book, output_dir)

        previous_dirs = server.BOOKS_DIRS
        server.BOOKS_DIRS = [self.temp_dir.name]
        server.clear_book_caches()
        try:
            client = TestClient(server.app)
            reader = client.get("/read/sample_data?page=2")
            source = client.get("/read/sample_data/source.pdf")
        finally:
            server.BOOKS_DIRS = previous_dirs
            server.clear_book_caches()

        self.assertEqual(reader.status_code, 200)
        self.assertIn('id="pdf-page-host"', reader.text)
        self.assertIn("Page 2 / 2", reader.text)
        self.assertIn("pdf.min.mjs", reader.text)
        self.assertEqual(source.status_code, 200)
        self.assertEqual(source.headers["content-type"], "application/pdf")
        self.assertTrue(source.content.startswith(b"%PDF-"))

    def test_image_only_pdf_gets_a_selectable_ocr_layer(self):
        visual_document = pymupdf.open()
        visual_page = visual_document.new_page(width=612, height=792)
        visual_page.insert_text((80, 180), "SELECTABLE OCR VOCABULARY", fontsize=28)
        pixmap = visual_page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        visual_document.close()

        scanned_path = os.path.join(self.temp_dir.name, "scanned.pdf")
        scanned_document = pymupdf.open()
        scanned_page = scanned_document.new_page(width=612, height=792)
        scanned_page.insert_image(scanned_page.rect, stream=pixmap.tobytes("png"))
        scanned_document.save(scanned_path)
        scanned_document.close()

        output_dir = os.path.join(self.temp_dir.name, "scanned_data")
        book = process_pdf(scanned_path, output_dir)
        searchable_document = pymupdf.open(os.path.join(output_dir, "scanned.pdf"))
        try:
            searchable_text = searchable_document[0].get_text("text")
        finally:
            searchable_document.close()

        self.assertEqual(book.ocr_page_count, 1)
        self.assertIn("SELECTABLE", searchable_text.upper())
        self.assertIn("VOCABULARY", searchable_text.upper())

    def test_old_pickles_default_to_epub_format(self):
        self.assertEqual(server.book_format(SimpleNamespace()), "epub")


if __name__ == "__main__":
    unittest.main()
