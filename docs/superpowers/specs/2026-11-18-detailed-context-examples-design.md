# Detailed Context Meaning and Examples Design

## Intent

Make new vocabulary explanations more useful for learning: the context meaning should explain the selected sense with a little more depth, and the flashcard back should show three natural examples of that same sense.

## Goals

- Keep the IPA pronunciation block.
- Expand Context meaning to 2–4 concise sentences covering the selected sense, its nuance, and why it fits the source sentence.
- Generate exactly three new, natural example sentences using the same sense.
- Display those examples on the flashcard back using the existing card content styles.
- Keep legacy two- and three-block explanations readable.

## Non-goals

- No new visual treatment or interaction design.
- No database schema change or automatic regeneration of existing explanations.
- No change to the reader tooltip or CSV layout; both continue using Context meaning only.

## Output Contract

The model returns exactly three `meaning-block` HTML fragments in this order:

1. An untitled IPA block.
2. A `Context meaning` block whose `meaning-text` contains 2–4 sentences.
3. An `Examples` block containing one `meaning-list` with exactly three `<li>` sentences.

The examples must use the same meaning as the source sentence, include the selected word or phrase naturally, and be distinct from the source sentence and one another.

## Approaches

- Chosen: extend the structured HTML contract and add a small `extract_examples()` parser. This matches the existing storage model, requires no migration, and allows safe template rendering.
- Rejected: change only the prompt. The current flashcard template deliberately renders parsed fields, so generated examples would remain hidden.
- Rejected: add database columns for context and examples. This duplicates data already stored in `ai_explanation` and adds unnecessary migration work.

## Compatibility

Missing or malformed Examples blocks produce an empty list. Older cards continue showing IPA and Context meaning without an empty Examples section.
