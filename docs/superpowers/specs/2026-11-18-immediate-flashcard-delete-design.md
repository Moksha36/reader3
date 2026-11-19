# Immediate Flashcard Delete Design

## Intent

Deleting a flashcard should take one action: pressing **Delete** immediately submits the existing delete form without opening a browser confirmation dialog.

## Scope

- Remove only the flashcard page's JavaScript submit-confirmation handler.
- Keep the existing `POST /vocabulary/delete` endpoint, word ID, return URL, and redirect behavior unchanged.
- Keep book deletion confirmations unchanged.
- Do not add undo, soft deletion, or new visual design.

## Approach

- Chosen: remove the `deleteForm` JavaScript binding and submit listener from `templates/flashcards.html`. This is the smallest change and preserves normal HTML form submission.
- Rejected: automatically accept or override `window.confirm`; this would retain unnecessary code and could affect unrelated confirmations.
- Rejected: change the server endpoint; the extra confirmation exists only in the browser, so server changes would add risk without changing the behavior.

## Verification

Render the flashcard template and verify the Delete form still targets `/vocabulary/delete`, while the page script no longer accesses `window.confirm` or installs a delete-form submit handler.
