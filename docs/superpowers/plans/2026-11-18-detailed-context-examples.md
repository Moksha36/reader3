# Detailed Context Meaning and Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate fuller context explanations and exactly three same-sense example sentences, then show those examples on flashcard backs.

**Architecture:** Extend the existing HTML explanation contract from two blocks to three. Parse example list items server-side into a safe `list[str]`, pass them with each flashcard entry, and render them using the existing meaning block/list markup.

**Tech Stack:** Python 3.10, FastAPI, BeautifulSoup, Jinja2, `unittest`

**Spec:** `docs/superpowers/specs/2026-11-18-detailed-context-examples-design.md`

## Global Constraints

- Model output is HTML fragments only and contains exactly three `meaning-block` elements.
- Context meaning is 2–4 concise sentences specific to the supplied source sentence.
- Examples contains exactly three distinct, natural sentences using the same contextual sense.
- Do not add database columns, dependencies, CSV columns, or new visual styling.
- Legacy explanations without an Examples block remain valid and render no empty examples section.

## Review Focus

- A legacy explanation without Examples returns `[]` and renders normally.
- An Examples block with extra non-list text extracts only direct list items.
- HTML inside an example is converted to normalized plain text before rendering.
- A selected phrase is preserved naturally in every generated example, not reduced to one word.
- The reader tooltip and CSV continue exposing Context meaning only.

---

### Task 1: Extend the AI explanation contract and parser

**Files:**
- Modify: `server.py:61-82,217-249`
- Modify: `tests/test_vocabulary_explanations.py:9-63`

**Interfaces:**
- Consumes: `translate_word_with_openai(word: str, sentence: str) -> str` and stored `ai_explanation` HTML.
- Produces: `extract_examples(ai_explanation: str) -> list[str]` for flashcard construction.

- [ ] **Step 1: Write failing prompt and parser tests**

Update the prompt test to assert exactly three `meaning-block` strings, the exact `Examples` title, explicit 2–4 sentence guidance, exactly three `<li>` elements, same-sense guidance, and no `General meaning`. Add parser tests asserting three normalized strings from a new explanation and `[]` from `LEGACY_EXPLANATION`; include inline HTML and non-list text in the fixture to pin plain-text/direct-list extraction.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `uv run --cache-dir .uv-cache python -m unittest tests.test_vocabulary_explanations.VocabularyExplanationTests -v`

Expected: FAIL because the prompt has two blocks and `extract_examples` does not exist.

- [ ] **Step 3: Implement the extended prompt and parser**

Update `VOCABULARY_EXPLANATION_SYSTEM_PROMPT` to the three-block contract in the spec. Add `extract_examples(ai_explanation: str) -> list[str]`, locating the block titled exactly `Examples` case-insensitively and returning normalized text from its direct `ol.meaning-list > li` children.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `uv run --cache-dir .uv-cache python -m unittest tests.test_vocabulary_explanations.VocabularyExplanationTests -v`

Expected: all VocabularyExplanationTests pass.

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_vocabulary_explanations.py
git commit -m "feat: generate detailed vocabulary examples"
```

### Task 2: Render generated examples on flashcard backs

**Files:**
- Modify: `server.py:1326-1347`
- Modify: `templates/flashcards.html:751-770`
- Modify: `tests/test_vocabulary_templates.py:14-67`

**Interfaces:**
- Consumes: `extract_examples(ai_explanation: str) -> list[str]` from Task 1.
- Produces: flashcard entry field `examples: list[str]`, consumed by `flashcards.html`.

- [ ] **Step 1: Write the failing flashcard rendering tests**

Give the fixture three examples, then assert the back contains one `Examples` title and exactly three `.meaning-list > li` values. Also render a fixture with `examples=[]` and assert no Examples title/list appears while IPA and Context meaning remain.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `uv run --cache-dir .uv-cache python -m unittest tests.test_vocabulary_templates.FlashcardTemplateTests -v`

Expected: FAIL because the template does not render `entry.examples`.

- [ ] **Step 3: Pass and render the examples**

Add `"examples": extract_examples(row[6])` when building flashcard dictionaries. In the existing `.card-back-content`, render a normal `meaning-block` titled `Examples` with an ordered `meaning-list` only when `entry.examples` is non-empty; do not add CSS.

- [ ] **Step 4: Run focused and compatibility tests**

Run: `uv run --cache-dir .uv-cache python -m unittest tests.test_vocabulary_templates tests.test_vocabulary_explanations -v`

Expected: all tests pass, including legacy and CSV assertions.

- [ ] **Step 5: Commit**

```bash
git add server.py templates/flashcards.html tests/test_vocabulary_templates.py
git commit -m "feat: show examples on flashcard backs"
```

### Task 3: Verify the complete feature branch

**Files:**
- Verify only.

**Interfaces:**
- Consumes: the completed explanation and flashcard changes.
- Produces: verification evidence for branch integration.

- [ ] **Step 1: Run the full automated suite**

Run: `uv run --cache-dir .uv-cache python -m unittest discover -s tests && node --test tests/*.js`

Expected: all Python and JavaScript tests pass.

- [ ] **Step 2: Check the diff**

Run: `git diff --check master...HEAD && git status --short`

Expected: no whitespace errors and no uncommitted product/test changes.

- [ ] **Step 3: Live-check a newly generated explanation**

Run the isolated app on port 8124, create a new word, and verify the saved response has IPA, a 2–4 sentence Context meaning, and exactly three Examples list items; confirm all three appear on the flashcard back and CSV still has eight columns with context in column 4 and source sentence in column 8.
