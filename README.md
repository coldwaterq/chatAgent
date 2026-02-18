# Chat Agent (NVIDIA Build + Z-Image-Turbo)

A simple Python chat agent: **chat** via [NVIDIA Build](https://build.nvidia.com), **image generation** via [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Hugging Face, runs locally with GPU). Run the web app to chat, generate images, and search the web.

## Quick start

```bash
pip install -r requirements.txt
```

Set `NVIDIA_API_KEY` in a `.env` file (copy from `.env.example`). Get a free key at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys).

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

- **Chat:** Type anything. The agent can answer, generate images (via the image tool), and search the web.
- Chat history is saved in `chat_history/`. Use the sidebar to switch between chats or start a **New chat**; sessions are restored when you reopen the app.
- Generated images are saved in `generated_images/` and shown in the UI (served at `/generated_images/...`). Hover an image to see the prompt and parameters used to create it.

## Models used

- **Chat:** `qwen/qwen3.5-397b-a17b` (streaming via NVIDIA integrate API). Requires `NVIDIA_API_KEY`.
- **Image generation:** Z-Image-Turbo (`Tongyi-MAI/Z-Image-Turbo`) via Hugging Face Diffusers, runs locally (CUDA recommended). No API key needed for images.

## API key

Copy `.env.example` to `.env` and set:

```env
NVIDIA_API_KEY=your_key_here
```

The app exits with instructions if the key is missing.

## Project layout

- **agent.py** — Shared library: NVIDIA config, tools (image gen, web search), `.env` loading.
- **app.py** — Web UI; uses `agent` and `requests` for streaming chat, persists sessions to `chat_history/`, serves images from `generated_images/`.

## Requirements

- Python 3.10+
- GPU with CUDA recommended for image generation (16GB+ VRAM for Z-Image-Turbo)
- `requirements.txt`: `requests`, `ddgs`, `flask`, `torch`, `diffusers` (from git), `transformers`, `accelerate`

For CUDA support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) first, then `pip install -r requirements.txt`.
