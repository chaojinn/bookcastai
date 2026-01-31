# BookcastAI

**Transform EPUB books into podcast-style audio feeds with AI-powered narration**

BookcastAI converts EPUB books into high-quality audio podcasts with automated text parsing, TTS generation, and RSS feed publishing. Access your favorite books as podcasts through an easy-to-use web interface or command-line automation.

## Key Features

- **Web-based interface** for easy book-to-audio conversion
- **EPUB parsing** with intelligent chapter extraction and text cleanup
- **AI-powered TTS generation** (Kokoro, OpenAI, DiaTTS)
- **Integrated audio player** with queue management
- **RSS podcast feed generation** for standard podcast apps
- **User authentication** and library management
- **GPU acceleration** for fast audio generation

---

## Quick Start (Docker - Recommended)

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with Docker GPU support (optional, for faster TTS)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bookcastai.git
   cd bookcastai
   ```

2. **Configure environment variables**

   Edit the `docker-compose.yml` file or create a `.env` file with these essential variables:

   ```env
   # Essential
   AUDIO_URL_PREFIX=https://your-domain.com
   OPENROUTE_API_KEY=sk-or-v1-your-key-here

   # Recommended
   OPENAI_API_KEY=sk-proj-your-key-here
   OLLAMA_API_URL=http://localhost:11434
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Access the web interface**

   Open your browser to `http://localhost:8000`

5. **Register an account**

   Navigate to the registration page and create your account

6. **Start converting books!**

   Upload an EPUB, parse it, generate audio, and enjoy

### Essential Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AUDIO_URL_PREFIX` | **Yes** | Public URL where audio files are accessible |
| `OPENROUTE_API_KEY` | **Yes** | OpenRouter API key for AI features |
| `OPENAI_API_KEY` | Recommended | OpenAI API key if using OpenAI TTS |
| `OLLAMA_API_URL` | Recommended | Ollama instance URL for chapter summaries |
| `PODS_BASE` | No | Book storage directory (default: `/app/data`) |

---

## Web Interface Guide

### Accessing the Application

After starting the Docker services, access the web interface at `http://localhost:8000`. First-time users should register an account at the registration page, then log in to access the full application.

### Converting a Book (Step-by-Step)

#### Step 1: Upload EPUB

1. Click the menu icon and select "Upload book"
2. Click "Choose File" and select your EPUB file
3. The folder name is auto-generated from the filename (editable)
4. If a folder with that name exists, you'll see a warning
5. Click "Upload" to begin
6. You'll be automatically redirected to the parsing page

#### Step 2: Parse EPUB

1. The parsing page displays your book title
2. (Optional) Enter CSS classes to ignore in the "Ignore CSS classes" field
   - Common examples: `footnote,page-number,toc,endnote`
3. Click "Parse EPUB" to start the extraction process
4. Watch the progress bar as chapters are analyzed
5. Once complete, you'll see a summary card showing:
   - Book title and chapter count
   - First and last chapter names
   - Average, longest, and shortest chapter word counts
   - Cover image (if available)
6. (Optional) Click "Toggle JSON Output" to view the raw parsed data
7. The page automatically redirects to TTS generation when ready

#### Step 3: Generate Audio (TTS)

1. **Select TTS Model**: Choose your preferred provider
   - **Kokoro** (default): Local GPU-accelerated, high quality
   - **OpenAI**: Cloud-based, requires API key
2. **Choose Voice**: Select from the dropdown menu
   - Kokoro offers multiple English voices (e.g., `af_heart`, `af_sarah`)
   - OpenAI offers voices like `alloy`, `echo`, `nova`, `shimmer`
3. **Adjust Speed**: Use the slider to set playback speed (0.6x - 1.2x)
   - Default: 1.0x (normal speed)
4. **Overwrite Option**: Check "Overwrite existing audio" to regenerate existing files
5. Click "Start TTS" to begin audio generation
6. Monitor progress as each chapter is converted
7. The RSS feed is automatically published upon completion

#### Step 4: Listen & Manage

1. Navigate to your **Library** to see all converted books
2. Click on a book to view its episode list (chapters)
3. Use the **integrated audio player** at the bottom of the interface
4. Queue management allows continuous playback across chapters
5. Your playback progress is saved per-user
6. The RSS feed is available at `{PODS_BASE}/{book_title}/book.xml` for use in standard podcast apps

### Library Management

- Browse your converted books with cover art thumbnails
- Click any book to view its chapters/episodes
- Track your listening progress automatically
- Delete or re-process books as needed

---

## Installation & Setup

### Docker Deployment (Recommended)

The `docker-compose.yml` file orchestrates three services:

1. **PostgreSQL** (port 5432) - Database for SuperTokens authentication
2. **SuperTokens Core** (port 3567) - Authentication service
3. **BookcastAI** (port 8000) - Main web application

#### Configuration

The docker-compose setup includes:
- **Volume mapping**: `./pod_data:/app/data` for persistent book storage
- **GPU support**: NVIDIA GPU acceleration for TTS (all GPUs available)
- **Networking**: All services communicate on the default bridge network

#### Managing Containers

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f bookcastai

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Local Development Setup

#### Prerequisites

- Python 3.12 or higher
- PostgreSQL 15+
- SuperTokens Core (can run via Docker)
- ffmpeg (for audio processing)
- Optional: NVIDIA GPU with CUDA for accelerated TTS

#### Installation Steps

1. **Clone and create virtual environment**
   ```bash
   git clone https://github.com/yourusername/bookcastai.git
   cd bookcastai
   python -m venv .venv
   ```

2. **Activate virtual environment**
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup PostgreSQL database**
   ```bash
   # Create database for SuperTokens
   createdb supertokens_db
   ```

5. **Start SuperTokens Core** (Docker recommended)
   ```bash
   docker run -p 3567:3567 -d \
     -e POSTGRESQL_CONNECTION_URI="postgresql://user:pass@localhost:5432/supertokens_db" \
     supertokens/supertokens-postgresql
   ```

6. **Configure environment variables**

   Create a `.env` file in the project root (see Environment Variables Reference below)

7. **Run the web server**
   ```bash
   python -m web.server
   ```

8. **Access the application**

   Open `http://localhost:8000` in your browser

---

## Environment Variables Reference

### Essential Configuration

- `PODS_BASE`: Base directory for book storage
  - Default: `./data` (or `/app/data` in Docker)
  - All book folders are created under this path
- `AUDIO_URL_PREFIX`: Public base URL where MP3 files are accessible
  - Example: `https://podcast.example.com`
  - Used to build RSS feed enclosure links

### AI & TTS Configuration

- `OPENROUTE_API_KEY` (or `OPENROUTER_API_KEY`): OpenRouter API key
  - Required for AI text cleanup (OCR artifact removal)
  - Required for chapter selection and organization
  - Required for cover image generation
- `OPENROUTE_MODEL`: AI model to use
  - Default: `google/gemini-2.5-flash-image-preview`
- `OPENROUTE_HTTP_REFERER`: HTTP referer header for OpenRouter API
- `OPENROUTE_HTTP_TITLE`: HTTP title header for OpenRouter API
- `OPENAI_API_KEY`: OpenAI API key
  - Required only if using OpenAI TTS provider
  - Model used: `gpt-4o-mini-tts`
- `OLLAMA_API_URL`: Ollama instance URL
  - Used for chapter summaries in RSS feed
  - Example: `http://localhost:11434`

### Authentication (SuperTokens)

- `SESSION_SECRET`: Secret key for session signing
  - Generate a secure random string
- `SUPERTOKENS_CORE_URL`: SuperTokens core service URL
  - Default: `http://localhost:3567`
  - In Docker: `http://supertokens:3567`
- `SUPERTOKENS_API_KEY`: API key for SuperTokens core
  - Must match the key configured in SuperTokens core
- `SUPERTOKENS_RECIPE_ID`: Authentication recipe
  - Default: `emailpassword`
- `SUPERTOKENS_CDI_VERSION`: Core Driver Interface version
  - Default: `4.0`
- `SUPERTOKENS_TENANT_ID`: Tenant ID
  - Default: `public`

### Database Configuration

- `DB_USER`: PostgreSQL username
- `DB_PASS`: PostgreSQL password
- `DB_HOST`: PostgreSQL host
  - Default: `localhost`
  - In Docker: `postgres`
- `DB_NAME`: PostgreSQL database name

### Remote Upload (Optional)

- `REMOTE_BASE`: Directory on remote host for uploads
  - Example: `/var/www/podcasts`
- `SCP_REMOTE` (or `REMOTE_HOST`): Remote host with optional port
  - Example: `podcast.example.com:22`
- `SCP_USERNAME`: SSH/SCP username
- `SCP_PASSWORD`: SSH/SCP password

### Example .env File

```env
# Essential
PODS_BASE=/app/data
AUDIO_URL_PREFIX=https://podcast.example.com

# AI & TTS
OPENROUTE_API_KEY=sk-or-v1-your-openrouter-key-here
OPENAI_API_KEY=sk-proj-your-openai-key-here
OLLAMA_API_URL=http://localhost:11434

# SuperTokens Authentication
SESSION_SECRET=your-secure-random-secret-here
SUPERTOKENS_CORE_URL=http://supertokens:3567
SUPERTOKENS_API_KEY=your-supertokens-api-key
SUPERTOKENS_RECIPE_ID=emailpassword
SUPERTOKENS_CDI_VERSION=4.0
SUPERTOKENS_TENANT_ID=public

# Database
DB_USER=admin
DB_PASS=your-secure-db-password
DB_HOST=postgres
DB_NAME=supertokens_db

# Optional: Remote Upload
REMOTE_BASE=/var/www/podcasts
SCP_REMOTE=podcast.example.com:22
SCP_USERNAME=deployer
SCP_PASSWORD=your-scp-password
```

---

## Architecture Overview

### Technology Stack

- **Backend**: FastAPI (Python 3.12+)
- **Frontend**: Bootstrap 5, jQuery, CodeMirror
- **Authentication**: SuperTokens (EmailPassword recipe)
- **Database**: PostgreSQL 15+
- **TTS Engines**: Kokoro, OpenAI TTS, DiaTTS
- **Audio Processing**: ffmpeg, pydub
- **AI**: OpenRouter (Gemini), Ollama (optional)
- **Containerization**: Docker with NVIDIA GPU support

### Directory Structure

```
bookcastai/
├── agent/              # EPUB parsing agent (LangGraph-based)
│   ├── epub_agent.py   # Main orchestration
│   ├── epub_mcp.py     # EPUB file interface
│   └── nodes/          # Graph processing nodes
├── tts/                # Text-to-speech provider implementations
│   ├── kokoro.py       # Kokoro TTS wrapper
│   ├── openai_provider.py  # OpenAI TTS wrapper
│   ├── dia_provider.py # Dia TTS wrapper
│   └── tts_provider.py # Base provider interface
├── web/                # FastAPI web application
│   ├── server.py       # FastAPI app setup
│   ├── PGDB.py         # Database wrapper
│   ├── html/           # Frontend HTML templates
│   ├── css/            # Stylesheets
│   ├── js/             # JavaScript (session management)
│   └── api/            # API route handlers
│       ├── job_queue.py    # Async job queue
│       ├── epub.py         # EPUB endpoints
│       ├── upload.py       # Upload endpoints
│       └── tts.py          # TTS endpoints
├── data/               # Default book storage (PODS_BASE)
├── bookcastai.py       # CLI pipeline script
├── epub_to_pod.py      # TTS generation script
├── feed.py             # RSS feed generator
├── docker-compose.yml  # Multi-service orchestration
├── Dockerfile          # Container image definition
└── requirements.txt    # Python dependencies
```

### Workflow Pipeline

1. **Upload**: EPUB file uploaded via web interface → stored as `{PODS_BASE}/{book_name}/book.epub`
2. **Parse**: EPUB agent extracts chapters, metadata, cover image → outputs `book.json`
3. **TTS**: Text-to-speech converts chapter text → MP3 files in `audio/` folder
4. **Feed**: RSS XML generated with enclosure links → `book.xml`
5. **Serve**: Audio files served via FastAPI static mount, RSS accessible to podcast apps

### Job Queue System

BookcastAI uses an asynchronous job queue for long-running tasks:
- **Commands**: `parse_epub`, `tts`
- **Real-time progress updates** via polling API endpoint `/api/job/{job_id}`
- **Status tracking**: `queued`, `in_progress`, `success`, `fail`, `cancelled`
- **Sequential processing**: One job at a time, with queue limited to 1000 jobs
- **Auto-pruning**: Oldest jobs removed when queue is full

---

## Advanced Usage

### Command-Line Interface (CLI Mode)

For automation, scripting, or server environments without web access, BookcastAI provides a command-line interface.

#### Full Pipeline

Run the complete pipeline (parse + TTS + feed) with a single command:

```bash
python bookcastai.py "book_title" [options]
```

**Parameters:**
- `book_title` (required): Folder name under `./data/` containing `book.epub`
- `--log-level`: Logging level (default: `INFO`)
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--chunk-size`: Maximum characters per text chunk (default: `200`)
- `--ignore-class`: CSS classes to ignore during parsing (repeatable)
  - Format: comma-separated values
  - Example: `footnote,page-number`
- `--ai-extract-text`: Enable AI text cleanup (slower, requires OpenRouter)
  - Uses AI to normalize OCR artifacts and improve TTS quality

**Examples:**

```bash
# Basic conversion
python bookcastai.py "my_book"

# With AI cleanup and custom chunk size
python bookcastai.py "my_book" --chunk-size 300 --ai-extract-text

# Ignore specific CSS classes
python bookcastai.py "my_book" --ignore-class footnote,page-number

# Debug mode with AI cleanup
python bookcastai.py "my_book" --log-level DEBUG --ai-extract-text

# Multiple ignore classes
python bookcastai.py "my_book" --ignore-class footnote --ignore-class endnote --ignore-class toc
```

#### Individual Scripts

Run pipeline steps separately for more control:

**1. Parse EPUB only**
```bash
python agent/epub_agent.py "book_title" --chunk-size 200 --ignore-class footnote
```

**2. Generate TTS only** (requires `book.json` to exist)
```bash
python epub_to_pod.py "book_title" --provider kokoro --voice af_sarah --speed 0.9
```

**3. Generate RSS feed only**
```bash
python feed.py "book_title"
```

#### CLI Output

Files are created in the following locations:
- **Parsed chapters**: `./data/{book_title}/book.json`
- **Audio files**: `./data/{book_title}/audio/*.mp3`
- **RSS feed**: `./data/{book_title}/book.xml`

### TTS Provider Configuration

#### Kokoro (Default)
- **Type**: Local GPU-accelerated TTS
- **Performance**: Fast, high quality
- **Requirements**: NVIDIA GPU with CUDA (recommended)
- **Voices**: Multiple English voices available
  - Examples: `af_heart`, `af_sarah`, `am_michael`
- **Advantages**: Offline, no API costs, fast with GPU

#### OpenAI TTS
- **Type**: Cloud-based TTS
- **Requirements**: `OPENAI_API_KEY` environment variable
- **Model**: `gpt-4o-mini-tts`
- **Voices**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- **Advantages**: No local GPU needed, consistent quality
- **CLI Usage**:
  ```bash
  python epub_to_pod.py "book_title" --provider openai --voice nova
  ```

#### DiaTTS (Experimental)
- **Type**: Advanced neural TTS
- **Custom parameter tuning** available
- **Consult source code** for advanced configuration options

### Custom Text Extraction

#### Ignoring CSS Classes

Use the `--ignore-class` flag to skip specific CSS classes during EPUB parsing. This is useful for removing unwanted content like footnotes, page numbers, or tables of contents.

**Common classes to ignore:**
- `footnote` - Footnote text
- `page-number` - Page numbers
- `toc` - Table of contents
- `endnote` - End notes
- `sidebar` - Sidebar content
- `caption` - Image captions

**Example:**
```bash
python bookcastai.py "my_book" --ignore-class footnote,page-number,toc
```

#### AI-Assisted Text Cleanup

Enable AI-powered text cleanup with the `--ai-extract-text` flag. This feature is particularly useful for:
- **OCR-scanned books** with formatting artifacts
- **Books with complex layouts** requiring normalization
- **Improving TTS quality** by cleaning up text

**Important notes:**
- Slower processing (each chunk sent to AI for cleanup)
- Requires `OPENROUTE_API_KEY` environment variable
- Uses OpenRouter API (costs apply)

**Example:**
```bash
python bookcastai.py "scanned_book" --ai-extract-text --chunk-size 300
```

---

## Troubleshooting

### Common Issues

**Problem:** Upload fails or parser can't find EPUB
- **Solution**: Check `PODS_BASE` environment variable is correctly set
- **Solution**: Ensure the application has write permissions to `PODS_BASE` directory
- **Solution**: Verify the EPUB file is not corrupted (try opening in an EPUB reader)

**Problem:** TTS generation is very slow
- **Solution**: Enable GPU acceleration - verify NVIDIA CUDA is properly configured
- **Solution**: In Docker, ensure GPU runtime is enabled in `docker-compose.yml`
- **Solution**: Check GPU usage with `nvidia-smi` during TTS generation
- **Solution**: Consider using OpenAI TTS provider if GPU is unavailable

**Problem:** Audio player doesn't show up or won't play
- **Solution**: Ensure the queue has items - click on a book and select chapters
- **Solution**: Check browser console (F12) for API errors
- **Solution**: Verify `AUDIO_URL_PREFIX` is correctly configured and accessible

**Problem:** SuperTokens authentication fails
- **Solution**: Verify SuperTokens Core is running: `curl http://localhost:3567/hello`
- **Solution**: Check `SUPERTOKENS_CORE_URL` matches actual SuperTokens service URL
- **Solution**: Ensure `SUPERTOKENS_API_KEY` matches between app and SuperTokens core
- **Solution**: Check database connection from SuperTokens to PostgreSQL

**Problem:** RSS feed links broken in podcast apps
- **Solution**: Ensure `AUDIO_URL_PREFIX` is publicly accessible (not `localhost`)
- **Solution**: Verify audio files are being served correctly - test direct MP3 URL
- **Solution**: Check firewall/network settings allow access to your server

**Problem:** "Job failed" error during parsing or TTS
- **Solution**: Check Docker logs: `docker-compose logs -f bookcastai`
- **Solution**: Verify all required API keys are set correctly
- **Solution**: Check available disk space in `PODS_BASE` directory

### Logs & Debugging

**Docker environment:**
```bash
# View real-time logs
docker-compose logs -f bookcastai

# View logs for all services
docker-compose logs -f

# Check specific service
docker-compose logs supertokens
```

**Job queue status:**
- Check progress bar in web interface
- API endpoint: `GET /api/job/{job_id}` returns status and progress

**CLI debug logging:**
```bash
python bookcastai.py "book_title" --log-level DEBUG
```

### GPU Support

**Verify NVIDIA Docker runtime:**
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Check GPU configuration in docker-compose.yml:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**For local development:**
- Install NVIDIA CUDA Toolkit
- Install PyTorch with CUDA support
- Verify GPU availability in Python:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

---

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

## License

[Specify your license here]

---

## Acknowledgments

- **Kokoro TTS** - High-quality local text-to-speech
- **OpenAI** - Cloud TTS services
- **OpenRouter** - AI API aggregation
- **SuperTokens** - Authentication framework
- **FastAPI** - Modern Python web framework
- **Libraries**: PyMuPDF, ebooklib, pydub, LangGraph, and many others
