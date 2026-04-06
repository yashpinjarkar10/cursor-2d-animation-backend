---
title: Manim Video Generator Backend
sdk: docker
app_port: 7860
---

# Manim Video Generator Backend

A FastAPI service that turns a text prompt into an educational Manim animation. Under the hood it uses a LangGraph workflow with Gemini models and (optionally) a ChromaDB RAG index built from Manim documentation.

## Deploy on Hugging Face Spaces (Docker)

1. Create a new Space → choose **Docker**.
2. In the Space settings, add a **Secret** named `GOOGLE_API_KEY`.
3. Ensure this backend folder is the Space repo root (it must contain `Dockerfile`, `app.py`, `requirements.txt`, etc.).
4. After the build finishes, open:
    - `https://<your-space>.hf.space/docs` (Swagger UI)
    - `https://<your-space>.hf.space/` (health/info)

## Features

- **Text → Video**: `POST /generate` returns an `.mp4` file directly.
- **LangGraph pipeline**: story → syntax questions → RAG → code → execute → (optional) single fix attempt.
- **RAG-backed Manim knowledge**: uses `chroma_db_manim/` (if present) to ground syntax and API usage.
- **Deterministic-ish codegen**: separate Gemini models for fast planning vs. code generation.
- **Editable + re-renderable**: `POST /render` executes user-supplied code and returns an `.mp4`.

## Technology Stack

- **API**: FastAPI
- **Orchestration**: LangGraph
- **LLM**: Google Gemini via `langchain-google-genai`
- **RAG**: ChromaDB + HuggingFace sentence-transformer embeddings
- **Rendering**: Manim Community Edition

## Project Structure

```
.
├── app.py              # FastAPI server + LangGraph workflow
├── prompts.py          # System prompts for each pipeline step
├── chroma_db_manim/    # ChromaDB vector store (Manim docs)
├── generated_videos/   # Output videos + saved generated code
├── requirements.txt    # Python dependencies (pip)
├── pyproject.toml      # Project metadata (optional/uv)
└── Dockerfile          # Containerized deployment
```

## Setup

### Prerequisites

- Python 3.10+
- A working Manim installation (Manim depends on system packages like FFmpeg + Cairo/Pango + LaTeX)
- A Google Gemini API key

Tip: if Manim dependencies are painful on your OS, use Docker (see below).

### 1) Install dependencies

```bash
python -m venv .venv

# Windows (PowerShell)
./.venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` file next to `app.py`:

```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

### 3) Run the API

```bash
python app.py
```

- Local dev server: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

## The AI Pipeline (LangGraph)

The backend uses a stateful graph to process user requests. This ensures a robust and debuggable workflow.

1.  **Generate Story**: The initial query is expanded into a visual narrative, breaking down the animation into distinct phases and describing visual elements.
2.  **Generate Syntax Questions**: The story is analyzed to create specific, technical questions about Manim syntax needed for implementation (e.g., "How to use `Transform` to change one shape into another?").
3.  **RAG Search**: The generated questions are used to search the ChromaDB vector store, which contains the Manim documentation. This retrieves relevant code snippets and explanations.
4.  **Generate Code**: The story, RAG search results, and original query are passed to the code generation LLM, which produces a complete, executable Manim Python script.
5.  **Execute Manim**: The generated script is executed using a `subprocess` call to Manim to render the video.
6.  **Review & Fix Code (Conditional Edge)**: If execution fails, the error and code are sent back to the LLM for a single fix attempt, which is executed once.

## Usage

### API Endpoints
FastAPI docs are available at `/docs`. Main endpoints:

| Method | Endpoint | Description |
| :-- | :-- | :-- |
| `POST` | `/generate` | Generate a video from a text query (returns `video/mp4`). |
| `POST` | `/render` | Render a video from a provided code string (returns `video/mp4`). |
| `GET` | `/get_code/{filename}` | Fetch a previously saved generated code file as JSON. |
| `GET` | `/` | Health/info + ChromaDB status. |

#### Example: generate from a prompt

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "Animate the process of binary search"}' \
  --output animation.mp4
```

Notes:
- `/generate` returns the MP4 directly (not JSON).
- The response includes useful headers like `X-Code-File-Path` pointing to the saved `.py` file under `generated_videos/`.

#### Example: render from code

```bash
curl -X POST "http://localhost:8000/render" \
    -H "Content-Type: application/json" \
    -d '{
        "filename": "my_scene",
        "SceneName": "Scene1",
        "code": "from manim import *\nfrom math import *\n\nclass Scene1(Scene):\n    def construct(self):\n        t = Text(\"Hello Manim\").to_edge(UP)\n        self.play(Write(t))\n        self.wait(1)\n"
    }' \
    --output render.mp4
```

### Output files

- Videos are copied to `generated_videos/animation_<tmpname>.mp4`.
- Code is saved to `generated_videos/generated_code_<tmpname>.py`.
- Manim’s intermediate outputs are written under `media/`.

### RAG (ChromaDB) behavior

- If `chroma_db_manim/` loads successfully, the service retrieves Manim docs snippets for the generated syntax questions.
- If it cannot be loaded, the service still runs, but code generation uses general Manim knowledge (you’ll see a startup warning and `chromadb_status: not available` at `/`).

## Docker

The container exposes port `7860`.

```bash
docker build -t manim-generator .
docker run --rm -p 7860:7860 -e GOOGLE_API_KEY="YOUR_GEMINI_API_KEY" manim-generator
```

Open http://localhost:7860/docs

## Contributing

Contributions are welcome! Please fork the repository, create a new feature branch, and submit a pull request. Make sure to follow the existing code style and add tests for any new functionality.
