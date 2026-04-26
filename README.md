# reader 3

![reader3](reader3.png)

A lightweight, self-hosted EPUB reader that lets you read through EPUB books one chapter at a time. This makes it very easy to copy paste the contents of a chapter to an LLM, to read along. Basically - get epub books (e.g. [Project Gutenberg](https://www.gutenberg.org/) has many), open them up in this reader, copy paste text around to your favorite LLM, and read together and along.

This project was 90% vibe coded just to illustrate how one can very easily [read books together with LLMs](https://x.com/karpathy/status/1990577951671509438). I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Usage

The project uses [uv](https://docs.astral.sh/uv/). So for example, download [Dracula EPUB3](https://www.gutenberg.org/ebooks/345) to this directory as `dracula.epub`, then:

```bash
uv run reader3.py dracula.epub
```

This creates the directory `dracula_data`, which registers the book to your local library. We can then run the server:

```bash
uv run server.py
```

And visit [localhost:8123](http://localhost:8123/) to see your current Library. You can easily add more books, or delete them from your library by deleting the folder. It's not supposed to be complicated or complex.

## OpenAI word translation

When you select one or more words in the reader and click `New Words`, the app can now ask OpenAI for explanations and save them into the vocabulary list. Select a single word to save it individually, or select multiple words to save the entire phrase as one entry.

1. Install the updated dependencies:

```bash
uv sync
```

2. Create an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

3. Put your key into `.env` in the project root:

```env
OPENAI_API_KEY=your_api_key_here
READER3_TRANSLATION_TARGET_LANGUAGE=English
READER3_OPENAI_MODEL=gpt-5-mini
```

4. Start the server:

```bash
uv run server.py
```

The app now auto-loads `.env`. If `OPENAI_API_KEY` is missing, it still works and saves the words, but skips the translation.

## License

MIT

- How to enter the program?
cd /Users/charliem/VSCode/github_projects/reader3
uv run server.py

then open http://127.0.0.1:8123

Delete all of the words in database:
psql postgresql:///reader3 -c "TRUNCATE TABLE new_words RESTART IDENTITY;"


- retart:
cd /Users/charliem/VSCode/github_projects/reader3
uv sync
DATABASE_URL="postgresql:///reader3" uv run python server.py


- how to add a book
