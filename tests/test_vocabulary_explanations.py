import csv
import io
import unittest
from types import SimpleNamespace

import server


LEGACY_EXPLANATION = """
<div class="meaning-block">
  <div class="meaning-text">/ɪɡˈzæmpəl/</div>
</div>
<div class="meaning-block">
  <div class="meaning-title">Context meaning</div>
  <div class="meaning-text">A representative case used to explain an idea.</div>
</div>
<div class="meaning-block">
  <div class="meaning-title">General meaning</div>
  <ol class="meaning-list"><li>A thing characteristic of its kind.</li></ol>
</div>
"""


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="generated explanation")


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


class VocabularyExplanationTests(unittest.TestCase):
    def test_translation_requests_only_pronunciation_and_context_blocks(self):
        fake_client = FakeOpenAIClient()
        previous_client = server.openai_client
        server.openai_client = fake_client
        try:
            result = server.translate_word_with_openai(
                "example",
                "This is an example sentence.",
            )
        finally:
            server.openai_client = previous_client

        system_prompt = fake_client.responses.request["input"][0]["content"]
        self.assertEqual(result, "generated explanation")
        self.assertEqual(system_prompt.count('<div class="meaning-block">'), 2)
        self.assertIn("Context meaning", system_prompt)
        self.assertNotIn("General meaning", system_prompt)

    def test_legacy_explanation_keeps_pronunciation_and_context_available(self):
        self.assertEqual(server.extract_pronunciation(LEGACY_EXPLANATION), "/ɪɡˈzæmpəl/")
        self.assertEqual(
            server.extract_context_meaning(LEGACY_EXPLANATION),
            "A representative case used to explain an idea.",
        )


class VocabularyExportTests(unittest.TestCase):
    def test_export_header_and_row_use_context_column_four_and_sentence_column_eight(self):
        expected_header = ["word", "", "", "context meaning", "", "", "", "sentence"]
        sentence = 'The author said, "This is an example."\nThen continued.'

        self.assertEqual(
            getattr(server, "VOCABULARY_EXPORT_HEADER", None),
            expected_header,
        )
        try:
            row = server.vocabulary_export_row(
                word="example",
                sentence=sentence,
                ai_explanation=LEGACY_EXPLANATION,
            )
        except TypeError as exc:
            self.fail(f"vocabulary_export_row must accept the source sentence: {exc}")

        self.assertEqual(len(row), 8)
        self.assertEqual(row[0], "example")
        self.assertEqual(row[3], "A representative case used to explain an idea.")
        self.assertEqual(row[7], sentence)
        self.assertNotIn("/ɪɡˈzæmpəl/", row)

        response = server.csv_response([expected_header, row], "vocabulary.csv")
        decoded_rows = list(csv.reader(io.StringIO(response.body.decode("utf-8-sig"))))
        self.assertEqual(decoded_rows[1][7], sentence)
        self.assertEqual(len(decoded_rows[1]), 8)


if __name__ == "__main__":
    unittest.main()
