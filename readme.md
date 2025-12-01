## summary
bookcastai turns an EPUB into a podcast-style feed: it parses the book, optionally cleans text with an AI pass, renders narration with TTS, and writes an RSS feed under `./data/{book_title}`.

## environment (.env)
- Create a `.env` in the repo root (loaded via `python-dotenv`) with these variables:
  - `OPENROUTE_API_KEY` (or `OPENROUTER_API_KEY`): OpenRouter key for AI text cleanup, chapter selection, and cover generation. Optional tuning: `OPENROUTE_MODEL` (defaults to `google/gemini-2.5-flash-image-preview`), `OPENROUTE_HTTP_REFERER`, `OPENROUTE_HTTP_TITLE`.
  - `OPENAI_API_KEY`: Needed only if you pick the OpenAI TTS provider (`--provider openai` in `epub_to_pod.py`).
  - `AUDIO_URL_PREFIX`: Public base URL where MP3s will be served (used to build RSS enclosure links).
  - `OLLAMA_API_URL`: URL to your Ollama instance (used for chapter summaries in the feed).
  - `REMOTE_BASE`: Directory on the remote host where feeds and audio are uploaded (e.g., `/var/www/podcasts`).
  - `SCP_REMOTE` (or `REMOTE_HOST`): `host` or `host:port` for uploads.
  - `SCP_USERNAME` / `SCP_PASSWORD`: Credentials used by `sshpass` for SCP/SSH during upload.
- Example template (fill with your own values):
  ```env
  OPENROUTE_API_KEY=your-openrouter-key
  OPENAI_API_KEY=your-openai-key
  AUDIO_URL_PREFIX=https://pod.example.com
  OLLAMA_API_URL=http://localhost:11434
  REMOTE_BASE=/var/www/podcasts
  SCP_REMOTE=pod.example.com:22
  SCP_USERNAME=deployer
  SCP_PASSWORD=choose-a-secure-password
  ```

## workflow
- Run the end-to-end pipeline with `python bookcastai.py "<book_title>"`. This calls `agent/epub_agent.py`, `epub_to_pod.py`, and `feed.py` in sequence so you get metadata, audio, and the finished RSS feed.
- Parameters for `bookcastai.py`:
  - `book_title` (positional): Folder name under `./data/` that contains the EPUB and where outputs are written.
  - `--log-level` (default `INFO`): Logging level passed to the EPUB agent.
  - `--chunk-size` (default `200`): Max characters per text chunk stored alongside the source text.
  - `--ignore-class` (repeatable): Comma-separated CSS classes to skip when extracting text; supply multiple times for more classes.
  - `--ai-extract-text`: Enable AI cleanup for OCR-like artifacts before TTS (requires OpenRouter credentials; slower).
- Example: `python bookcastai.py "my_book" --chunk-size 300 --ignore-class note,footnote --ai-extract-text`
- Outputs: cleaned chapter chunks, generated audio, and an RSS feed under `./data/<book_title>/` ready to load into a podcast player.
   
