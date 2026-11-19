# Immediate Flashcard Delete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the flashcard Delete button submit immediately without a browser confirmation dialog.

**Architecture:** Keep the existing HTML delete form and server endpoint. Remove only the flashcard page's client-side submit listener that calls `window.confirm`.

**Tech Stack:** Jinja2, JavaScript, Python `unittest`, BeautifulSoup

**Spec:** `docs/superpowers/specs/2026-11-18-immediate-flashcard-delete-design.md`

## Global Constraints

- Keep `POST /vocabulary/delete`, active word ID synchronization, return URL, and redirect behavior unchanged.
- Keep book deletion confirmations unchanged.
- Add no undo UI, soft deletion, dependencies, or visual changes.

## Review Focus

- The flashcard Delete form still posts to `/vocabulary/delete`.
- The active flashcard word ID remains the submitted ID.
- The current flashcard URL remains the post-delete return destination.
- No flashcard script opens a confirmation dialog or cancels deletion.
- Library book deletion confirmations remain unchanged.

---

### Task 1: Remove the flashcard delete confirmation

**Files:**
- Modify: `templates/flashcards.html:844-847,876-879,1068-1073`
- Test: `tests/test_vocabulary_templates.py`

**Interfaces:**
- Consumes: existing `POST /vocabulary/delete` form contract with `word_id` and `return_to`.
- Produces: immediate native form submission from the flashcard Delete button.

- [ ] **Step 1: Write the failing template behavior test**

Add `test_delete_submits_immediately_without_confirmation`. Render the flashcard template, assert the `#deleteForm` method/action and hidden fields remain correct, and assert its rendered scripts contain neither `window.confirm` nor a `deleteForm.addEventListener("submit"...)` binding. Retain the existing library template confirmation behavior unchanged.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `uv run --cache-dir .uv-cache python -m unittest tests.test_vocabulary_templates.FlashcardTemplateTests.test_delete_submits_immediately_without_confirmation -v`

Expected: FAIL because the rendered flashcard script still contains `window.confirm`.

- [ ] **Step 3: Remove only the confirmation listener**

In `templates/flashcards.html`, remove the unused `deleteForm` constant and its submit event listener. Keep `deleteWordId`, `syncWordIds()`, and the form markup unchanged.

- [ ] **Step 4: Run focused and full verification**

Run: `uv run --cache-dir .uv-cache python -m unittest tests.test_vocabulary_templates -v && uv run --cache-dir .uv-cache python -m unittest discover -s tests && node --test tests/*.js`

Expected: all template, Python, and JavaScript tests pass.

- [ ] **Step 5: Commit**

```bash
git add templates/flashcards.html tests/test_vocabulary_templates.py
git commit -m "feat: delete flashcards without confirmation"
```
