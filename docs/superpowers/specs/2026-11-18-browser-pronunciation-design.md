# Browser Pronunciation Design

## Intent

Add on-demand audible pronunciation to saved vocabulary in the reader tooltip and flashcard review. Speech should begin only after the user presses a speaker control, with no API charge, generated audio files, database migration, or new dependency.

## Behavior

- Use the browser Web Speech API (`window.speechSynthesis`) to pronounce the original saved word or phrase.
- Prefer a high-quality English voice reported as local by the browser, then fall back to any English voice, then the browser default.
- Speak at a slightly reduced rate for vocabulary study.
- Cancel existing speech before starting another pronunciation.
- Disable or hide pronunciation controls when speech synthesis is unavailable.
- Give each control an accessible label such as `Pronounce example` and a visible speaker icon.

## Explanation Content

- Change the OpenAI prompt so new explanations contain exactly two blocks: IPA pronunciation and context meaning.
- Remove the `General meaning` block and its list from future model output.
- Render only pronunciation and context meaning on flashcards, including for explanations that were saved before this change.
- Preserve existing database values rather than destructively rewriting saved vocabulary.

## CSV Export

- Keep the existing eight-column export shape.
- Remove pronunciation from exported rows.
- Put context meaning in column 4.
- Put the saved source sentence in column 8.
- Apply the same mapping to full-library, book-filtered, and chapter exports.

## Reader Tooltip

- Preserve the original saved word alongside its context meaning in the reader's client-side map.
- Add the word and a speaker button to the floating tooltip header.
- Allow pointer interaction inside the tooltip so the button can be pressed without immediately closing it.
- Keep the existing hover positioning and background explanation refresh behavior.

## Flashcards

- Add a speaker button to both card faces near the displayed word.
- Change the outer flashcard from a native `<button>` to a focusable interactive container so it can legally contain speaker buttons.
- Preserve click-to-flip and Space/Enter keyboard behavior on the card container.
- Stop propagation from speaker buttons so pronunciation never flips the card.
- Stop speech when moving to another card or leaving the page.

## Approaches Considered

- **Browser speech synthesis — selected:** immediate, free, offline-capable, and sufficient for short vocabulary. The exact voice depends on the user's browser and operating system.
- **OpenAI Realtime speech:** more consistent across devices, but adds API cost and streaming/backend complexity for a one-word action.
- **Pre-generated stored audio:** fastest repeat playback, but adds storage, cleanup, and unnecessary generation for words never played.

## Scope and Verification

No new server route, schema, or model change is required. The existing explanation prompt, flashcard presentation, and CSV column mapping will change. Verification will cover the two-block prompt, removal of general meanings from old and new flashcards, the eight-column CSV mapping, tooltip playback, both flashcard faces, no accidental card flips, keyboard controls, card navigation cancellation, unsupported-browser behavior, and regression tests for existing server behavior.
