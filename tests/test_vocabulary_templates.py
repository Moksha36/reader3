import unittest
from types import SimpleNamespace

from bs4 import BeautifulSoup

import server
from tests.test_vocabulary_explanations import LEGACY_EXPLANATION


class FlashcardTemplateTests(unittest.TestCase):
    def render_flashcards(self) -> BeautifulSoup:
        template = server.templates.env.get_template("flashcards.html")
        html = template.render(
            request=SimpleNamespace(
                url=SimpleNamespace(path="/vocabulary/flashcards", query="")
            ),
            flashcards=[{
                "id": 7,
                "book_id": "sample_data",
                "book_title": "Sample Book",
                "chapter_index": 0,
                "chapter_number": 1,
                "location_label": "Chapter 1",
                "page_number": 1,
                "word": "example",
                "sentence": "This is an example sentence.",
                "highlighted_sentence": "This is an <mark>example</mark> sentence.",
                "ai_explanation": LEGACY_EXPLANATION,
                "pronunciation": "/ɪɡˈzæmpəl/",
                "context_meaning": "A representative case used to explain an idea.",
                "memorized": False,
                "note": "",
                "created_at": None,
            }],
            book_id="sample_data",
            book_title="Sample Book",
            memorized_count=0,
            total_count=1,
            notes_by_id={"7": ""},
        )
        return BeautifulSoup(html, "html.parser")

    def test_card_has_two_pronunciation_controls_without_rendering_general_meaning(self):
        page = self.render_flashcards()
        card = page.select_one(".flashcard")
        controls = page.select(".pronounce-button")

        self.assertIsNotNone(card)
        self.assertEqual(card.name, "div")
        self.assertEqual(card.get("role"), "button")
        self.assertEqual(card.get("tabindex"), "0")
        self.assertEqual(len(controls), 2)
        self.assertEqual(
            [control.get("data-pronunciation") for control in controls],
            ["example", "example"],
        )
        self.assertTrue(all(control.get("aria-label") == "Pronounce example" for control in controls))
        self.assertIn("/ɪɡˈzæmpəl/", page.get_text(" ", strip=True))
        self.assertIn(
            "A representative case used to explain an idea.",
            page.get_text(" ", strip=True),
        )
        self.assertNotIn("General meaning", page.get_text(" ", strip=True))


if __name__ == "__main__":
    unittest.main()
