# Browser Pronunciation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add on-demand browser-generated pronunciation to reader tooltips and both flashcard faces, remove general meanings from new and existing flashcard presentation, and revise the eight-column CSV export.

**Architecture:** Keep speech entirely client-side behind a small `window.Reader3Speech` module that wraps the Web Speech API. Refine the server's explanation prompt to two blocks, use the existing HTML extractors to render only pronunciation and context meaning while preserving old database values, and centralize the revised CSV column mapping in one row helper.

**Tech Stack:** FastAPI, Jinja2, vanilla JavaScript, Web Speech API, Python `unittest`, Node.js built-in test runner.

**Spec:** `docs/superpowers/specs/2026-11-18-browser-pronunciation-design.md`

## Global Constraints

- No new Python or JavaScript dependency.
- No database migration and no destructive rewrite of existing `ai_explanation` values.
- Speech starts only after an explicit user action.
- Pronunciation controls must be keyboard accessible and must not flip flashcards.
- New AI explanations contain only IPA pronunciation and context meaning.
- CSV exports omit pronunciation, put context meaning in column 4, and put the saved source sentence in column 8.

## Review Focus

- Browser exposes `speechSynthesis` but returns no voices initially: speech still uses the default voice and remains usable.
- Empty or whitespace-only pronunciation input: no utterance is created.
- Speaker button receives Space or Enter: it speaks once and does not trigger the global flip shortcut.
- Moving to another flashcard while speech is active: current speech is cancelled.
- Old saved HTML still contains `General meaning`: flashcards render only pronunciation and context meaning without modifying stored data.
- Sentences containing commas, quotes, or line breaks: the CSV writer preserves the sentence as one correctly escaped column 8 value.

---

### Task 1: Two-block vocabulary explanations

**Files:**
- Modify: `server.py:289-359`
- Modify: `server.py:1341-1360`
- Create: `tests/test_vocabulary_explanations.py`

**Interfaces:**
- Consumes: existing `extract_pronunciation(ai_explanation: str) -> str` and `extract_context_meaning(ai_explanation: str) -> str`.
- Produces: `VOCABULARY_EXPLANATION_SYSTEM_PROMPT: str`; flashcard dictionaries with `pronunciation: str` and `context_meaning: str`.

- [ ] **Step 1: Write failing prompt and legacy-explanation tests**

Add tests asserting the prompt requires exactly two blocks, contains `Context meaning`, contains no `General meaning`, and the existing extractors recover IPA and context from a legacy three-block explanation.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `uv run python -m unittest tests.test_vocabulary_explanations -v`

Expected: FAIL because the prompt is not exposed as a constant and still requests `General meaning`.

- [ ] **Step 3: Extract and reduce the prompt**

Define `VOCABULARY_EXPLANATION_SYSTEM_PROMPT` in `server.py`, require exactly the IPA and `Context meaning` blocks, and pass that constant to `openai_client.responses.create`.

- [ ] **Step 4: Supply presentation-safe flashcard fields**

Populate `pronunciation` and `context_meaning` in each flashcard dictionary using the existing extractors; keep `ai_explanation` only as stored source data, not as rendered HTML.

- [ ] **Step 5: Run the focused tests and verify pass**

Run: `uv run python -m unittest tests.test_vocabulary_explanations -v`

Expected: all tests PASS.

### Task 2: Shared browser speech module

**Files:**
- Create: `static/speech.js`
- Create: `tests/test_speech.js`

**Interfaces:**
- Consumes: browser globals `window.speechSynthesis` and `window.SpeechSynthesisUtterance`.
- Produces: `window.Reader3Speech.isSupported() -> boolean`, `speak(text: string) -> boolean`, and `stop() -> void`.

- [ ] **Step 1: Write failing Web Speech wrapper tests**

Use Node's `node:test` and `vm` modules with stub browser globals. Assert unsupported environments return `false`, blank text creates no utterance, `speak` cancels existing speech, prefers a local English voice, sets `lang` to `en-US`, uses rate `0.85`, and `stop` cancels speech.

- [ ] **Step 2: Run the JavaScript tests and verify failure**

Run: `node --test tests/test_speech.js`

Expected: FAIL because `static/speech.js` does not exist.

- [ ] **Step 3: Implement the shared speech module**

Create an IIFE that exposes only the three documented methods, selects local English then any English voice, cancels before speaking, and safely returns `false` when unsupported or given blank input.

- [ ] **Step 4: Run the JavaScript tests and verify pass**

Run: `node --test tests/test_speech.js`

Expected: all tests PASS.

### Task 3: Flashcard pronunciation controls

**Files:**
- Modify: `templates/flashcards.html:1-980`
- Create: `tests/test_vocabulary_templates.py`

**Interfaces:**
- Consumes: `window.Reader3Speech`; flashcard fields `word`, `pronunciation`, and `context_meaning` from Task 1.
- Produces: two `.pronounce-button` controls per card with `data-pronunciation`; a focusable `.flashcard[role="button"]` container.

- [ ] **Step 1: Write a failing rendered-template structure test**

Render `flashcards.html` with one legacy explanation and assert: two speaker buttons exist; both carry the original word/phrase; the card is not a native button; the output includes IPA and context meaning; and `General meaning` is absent.

- [ ] **Step 2: Run the template test and verify failure**

Run: `uv run python -m unittest tests.test_vocabulary_templates.FlashcardTemplateTests -v`

Expected: FAIL because the existing template renders raw explanation HTML and has no speaker controls.

- [ ] **Step 3: Refactor the card container and explanation markup**

Load `/static/speech.js`; change the outer card to a focusable `div` with `role="button"`; add styled speaker buttons beside the word on both faces; render explicit pronunciation and context blocks rather than `ai_explanation | safe`.

- [ ] **Step 4: Wire flashcard speech without flip side effects**

On speaker click and key activation, stop propagation and call `Reader3Speech.speak`. Ignore the global Space/Enter flip shortcut when the event target is a button. Stop speech on card navigation and `pagehide`.

- [ ] **Step 5: Run the template and speech tests**

Run: `uv run python -m unittest tests.test_vocabulary_templates.FlashcardTemplateTests -v && node --test tests/test_speech.js`

Expected: all tests PASS.

### Task 4: Reader tooltip pronunciation control

**Files:**
- Modify: `templates/reader.html:341-377`
- Modify: `templates/reader.html:621-650`
- Modify: `templates/reader.html:951-1126`
- Modify: `templates/reader.html:1320-1343`
- Modify: `tests/test_vocabulary_templates.py`

**Interfaces:**
- Consumes: `window.Reader3Speech`; saved-word API entries containing `word` and `ai_explanation`.
- Produces: `savedWordEntries: Map<string, {word: string, meaning: string}>`; an interactive tooltip speaker control.

- [ ] **Step 1: Extend the failing template test for tooltip structure**

Assert `reader.html` loads `/static/speech.js`, contains a tooltip word element and accessible speaker button, and does not place the speaker button inside the highlighted word element.

- [ ] **Step 2: Run the reader template test and verify failure**

Run: `uv run python -m unittest tests.test_vocabulary_templates.ReaderTemplateTests -v`

Expected: FAIL because the tooltip currently contains only context text.

- [ ] **Step 3: Preserve word and meaning in reader state**

Replace the meaning-only map with `savedWordEntries`; update `rememberSavedWordEntry`, metadata refresh, background refresh, and tooltip display to use `{word, meaning}` entries.

- [ ] **Step 4: Add the interactive tooltip control**

Add tooltip header markup, speaker styling, click behavior, visible-state pointer events, pointer transitions between highlighted text and tooltip, and `pagehide` speech cancellation. Disable the control when Web Speech is unavailable.

- [ ] **Step 5: Run the focused template tests**

Run: `uv run python -m unittest tests.test_vocabulary_templates -v`

Expected: all tests PASS.

### Task 5: Revised CSV export mapping

**Files:**
- Modify: `server.py:228-238`
- Modify: `server.py:1462-1535`
- Modify: `tests/test_vocabulary_explanations.py`

**Interfaces:**
- Consumes: `extract_context_meaning(ai_explanation: str) -> str`; database rows containing `word`, `sentence`, and `ai_explanation`.
- Produces: `VOCABULARY_EXPORT_HEADER: list[str]`; `vocabulary_export_row(word: str, sentence: str, ai_explanation: str) -> list[str]` with exactly eight fields.

- [ ] **Step 1: Write failing CSV mapping tests**

Assert the header names column 4 `context meaning` and column 8 `sentence`; exported rows contain exactly eight fields; column 4 contains the parsed context meaning; column 8 contains the original sentence including commas and quotes; and pronunciation is absent.

- [ ] **Step 2: Run the CSV tests and verify failure**

Run: `uv run python -m unittest tests.test_vocabulary_explanations.VocabularyExportTests -v`

Expected: FAIL because the current row helper places pronunciation in column 4 and context meaning in column 8.

- [ ] **Step 3: Implement the eight-column row mapping**

Add `VOCABULARY_EXPORT_HEADER`, change the helper signature and mapping, select `sentence` in both full/book and chapter queries, and pass `(word, sentence, ai_explanation)` for every exported row.

- [ ] **Step 4: Run the CSV tests and verify pass**

Run: `uv run python -m unittest tests.test_vocabulary_explanations.VocabularyExportTests -v`

Expected: all tests PASS.

### Task 6: Full verification

**Files:**
- Verify only; no planned code changes.

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: verified feature behavior.

- [ ] **Step 1: Run the complete Python test suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: all tests PASS with zero failures and zero errors.

- [ ] **Step 2: Run the complete JavaScript test suite**

Run: `node --test tests/test_speech.js`

Expected: all tests PASS.

- [ ] **Step 3: Restart and smoke-test Reader3**

Start the server with `DATABASE_URL="postgresql:///reader3" uv run uvicorn server:app --host 127.0.0.1 --port 8123`, verify `/` returns HTTP 200, verify the new explanation prompt generates a response with IPA and context meaning but no general meaning, and inspect an exported row to confirm context meaning is column 4 and sentence is column 8.

- [ ] **Step 4: Verify browser interaction manually**

Check speaker playback in the reader tooltip and on both flashcard faces; confirm speaker clicks do not flip cards, Space/Enter still flips focused cards, navigation cancels speech, and the controls are unavailable rather than broken when Web Speech is unsupported.
