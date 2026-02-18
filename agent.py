"""
Chat agent library: NVIDIA Build chat config, Z-Image-Turbo (Hugging Face) for image gen, web search, and helpers.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _ROOT / ".env"


def _load_dotenv() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1].replace('\\"', '"')
            elif v.startswith("'") and v.endswith("'"):
                v = v[1:-1].replace("\\'", "'")
            os.environ.setdefault(k, v)


_load_dotenv()

import requests

# --- Config (NVIDIA Build for chat only) ---
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
CHAT_MODEL = "qwen/qwen3.5-397b-a17b"
OUTPUT_DIR = _ROOT / "generated_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Z-Image-Turbo model (Hugging Face); pipeline loaded lazily
Z_IMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
_pipeline: object = None

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text description. Use when the user asks for a new picture, illustration, or image. Optional: cfg_scale, aspect_ratio, steps, seed. Image editing is not available; for changes suggest a new prompt and generate again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed text description of the image to generate (e.g. 'a cozy coffee shop at sunset').",
                    },
                    "cfg_scale": {
                        "type": "number",
                        "description": "How closely to follow the prompt. Default 3.5.",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Output aspect ratio (e.g. '16:9', '1:1'). Default 16:9.",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of inference steps (Turbo model uses ~8–9). Default 9.",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility. Use 0 for random. Default 0.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Use when the user asks about recent events, facts, or anything you need to look up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'weather Tokyo', 'latest news about AI').",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# Z-Image-Turbo 1024-category resolutions (width, height).
_ASPECT_TO_SIZE: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "9:7": (1152, 896),
    "7:9": (896, 1152),
    "21:9": (1344, 576),
    "9:21": (576, 1344),
}


def _get_pipeline():
    """Load Z-Image-Turbo pipeline once and cache it. Uses CUDA if available."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        import torch
        from diffusers import ZImagePipeline
    except ImportError as e:
        raise RuntimeError(
            "Z-Image-Turbo requires torch and diffusers. "
            "Install with: pip install torch 'diffusers @ git+https://github.com/huggingface/diffusers.git'"
        ) from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log.info("Loading Z-Image-Turbo pipeline on %s (first run may download the model)...", device)
    _pipeline = ZImagePipeline.from_pretrained(
        Z_IMAGE_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    _pipeline.to(device)
    log.info("Z-Image-Turbo pipeline ready.")
    return _pipeline


def generate_image(
    api_key: str,
    prompt: str,
    *,
    cfg_scale: float | None = None,
    aspect_ratio: str | None = None,
    steps: int | None = None,
    seed: int | None = None,
) -> tuple[str, str | None, dict]:
    """Text-to-image via Z-Image-Turbo (Hugging Face). Returns (message_for_model, image_url_or_none, params_for_ui)."""
    import torch

    prompt_text = (prompt or "").strip()[:10000]
    ar = (aspect_ratio or "1:1").strip() or "1:1"
    width, height = _ASPECT_TO_SIZE.get(ar, (1024, 1024))
    st = steps if steps is not None else 9
    st = max(1, min(20, int(st)))
    sd = seed if seed is not None else 0
    if sd != 0:
        sd = int(sd)
    else:
        sd = random.randint(1, 2**31 - 1)

    log.info("Image create (Z-Image-Turbo) requested: prompt=%r", prompt_text[:200])

    try:
        pipe = _get_pipeline()
        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(sd)
        image = pipe(
            prompt=prompt_text,
            height=height,
            width=width,
            num_inference_steps=st,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        safe_name = re.sub(r"[^\w\s-]", "", prompt_text)[:50].strip() or "image"
        safe_name = re.sub(r"[-\s]+", "_", safe_name)
        unique = uuid.uuid4().hex[:8]
        path = OUTPUT_DIR / f"{safe_name}_{unique}.png"
        image.save(path)
        image_url = f"/generated_images/{path.name}"
        log.info("Image saved: %s", path.name)
        params = {
            "prompt": prompt_text,
            "seed": sd,
            "steps": st,
            "cfg_scale": 0.0,
            "aspect_ratio": ar,
            "width": width,
            "height": height,
        }
        return f"Image saved to: {path}", image_url, params
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            log.warning("Image generation failed (GPU): %s", e)
            return (
                "Image generation failed (GPU error or out of memory). "
                "Try again or use a smaller resolution.",
                None,
                {},
            )
        raise
    except Exception as e:
        log.warning("Image generation failed: %s", e, exc_info=True)
        return f"Image generation failed: {e}", None, {}


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, snippets, and URLs."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Web search unavailable. Run: pip install ddgs"
    query = (query or "").strip()[:500]
    if not query:
        return "No search query provided."
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        body = r.get("body") or ""
        href = r.get("href") or r.get("link") or ""
        lines.append(f"{i}. {title}\n   {body}\n   {href}")
    return "\n\n".join(lines)


def run_tool(name: str, args: dict, api_key: str) -> tuple[str, str | None, dict]:
    """Returns (message_for_model, image_url_or_none, params_for_ui). params_for_ui only set for generate_image."""
    if name == "generate_image":
        return generate_image(
            api_key,
            args.get("prompt", ""),
            cfg_scale=args.get("cfg_scale"),
            aspect_ratio=args.get("aspect_ratio"),
            steps=args.get("steps"),
            seed=args.get("seed"),
        )
    if name == "search_web":
        return search_web(args.get("query", "")), None, {}
    return f"Unknown tool: {name}", None, {}
