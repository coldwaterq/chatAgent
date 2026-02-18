#!/usr/bin/env python3
"""
Web UI for the chat agent. Uses agent library, persistent sessions on disk, streaming chat with tools.
Run: python app.py
"""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime

import requests

log = logging.getLogger(__name__)

from agent import TOOLS, OUTPUT_DIR, run_tool

_ROOT = Path(__file__).resolve().parent
HISTORY_DIR = _ROOT / "chat_history"
USER_IMAGES_DIR = _ROOT / "user_images"
HISTORY_DIR.mkdir(exist_ok=True)
USER_IMAGES_DIR.mkdir(exist_ok=True)

CHAT_MODEL = "qwen/qwen3.5-397b-a17b"
# NVIDIA chat API allows at most 8 images per message
MAX_IMAGES_PER_PROMPT = 8

SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to image generation and web search. "
    "For images use generate_image with a detailed prompt. Image editing is not available; if the user wants changes, suggest a new prompt and generate again. "
    "When the user asks about things you do not know, use the search_web tool to find relevant information."
)


def _messages_for_storage(messages: list[dict]) -> list[dict]:
    """Persist full conversation including tool_calls and tool results so the LLM sees them in history."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "content": m.get("content") or ""})
        elif role == "user":
            c = m.get("content")
            out.append({"role": "user", "content": c if c is not None else ""})
        elif role == "assistant":
            msg = {
                "role": "assistant",
                "content": m.get("content") or "",
                "images": m.get("images") or [],
            }
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            out.append(msg)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": m.get("content") or "",
            })
    return out


def _user_content_to_api(content, session_id: str) -> str | list[dict]:
    """Convert stored user message content to API format. Resolve /user_images/ paths to data URLs. Caps at MAX_IMAGES_PER_PROMPT images."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) or ""
    parts = []
    image_count = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            if part.get("text"):
                parts.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url" and image_count < MAX_IMAGES_PER_PROMPT:
            url = (part.get("image_url") or {}).get("url") or ""
            if url.startswith("/user_images/") and session_id:
                # Resolve path to data URL for API
                rel = url.replace("/user_images/", "").lstrip("/")
                path = USER_IMAGES_DIR / rel
                if path.exists():
                    try:
                        raw = path.read_bytes()
                        b64 = base64.b64encode(raw).decode("ascii")
                        mime = "image/jpeg"
                        if path.suffix.lower() in (".png",):
                            mime = "image/png"
                        elif path.suffix.lower() in (".gif", ".webp"):
                            mime = f"image/{path.suffix.lower()[1:]}"
                        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                        image_count += 1
                    except Exception:
                        pass
            elif url.startswith("data:"):
                parts.append({"type": "image_url", "image_url": {"url": url}})
                image_count += 1
    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0].get("text", "")
    return parts


def _generated_image_to_data_url(url: str) -> str | None:
    """Convert /generated_images/<filename> to a data URL so the LLM can see the image. Returns None if file missing."""
    if not url or not url.startswith("/generated_images/"):
        return None
    rel = url.replace("/generated_images/", "").lstrip("/")
    path = OUTPUT_DIR / rel
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        mime = "image/jpeg"
        if path.suffix.lower() in (".png",):
            mime = "image/png"
        elif path.suffix.lower() in (".gif", ".webp",):
            mime = f"image/{path.suffix.lower()[1:]}"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _messages_from_storage(stored: list[dict]) -> list[dict]:
    """Restore full history for API (including tool_calls and tool results)."""
    return list(stored)


def load_sessions() -> list[dict]:
    """Return list of { id, title, created_at, updated_at } sorted by updated_at desc. Excludes soft-deleted."""
    sessions = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("deleted"):
                continue
            sessions.append({
                "id": data.get("id", f.stem),
                "title": data.get("title", "New chat"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        except Exception:
            continue
    sessions.sort(key=lambda s: s["updated_at"] or s["created_at"] or "", reverse=True)
    return sessions


def load_session(session_id: str) -> dict | None:
    """Return { id, title, created_at, updated_at, messages } or None. Returns None if session is soft-deleted."""
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("deleted"):
            return None
        data["messages"] = _messages_from_storage(data.get("messages", []))
        return data
    except Exception:
        return None


def save_session(session_id: str, messages: list[dict], title: str | None = None) -> str:
    """Persist session; derive title from first user message if not set. Returns title."""
    path = HISTORY_DIR / f"{session_id}.json"
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        created_at = existing.get("created_at", now)
        if not title:
            title = existing.get("title", "New chat")
    else:
        created_at = now
        if not title:
            for m in messages:
                if m.get("role") != "user":
                    continue
                raw = m.get("content")
                if not raw:
                    continue
                if isinstance(raw, str):
                    title = (raw or "").strip()[:50] or "New chat"
                else:
                    text = ""
                    if isinstance(raw, list):
                        for part in raw:
                            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                                text = (part.get("text") or "").strip()
                                break
                    title = text[:50] if text else "New chat"
                if title:
                    break
            else:
                title = "New chat"
    to_save = {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": now,
        "messages": _messages_for_storage(messages),
    }
    path.write_text(json.dumps(to_save, indent=2), encoding="utf-8")
    return title


def update_session_title(session_id: str, title: str) -> bool:
    """Update a session's title. Returns True if updated."""
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["title"] = (title or "New chat").strip()[:200]
        data["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def soft_delete_session(session_id: str) -> bool:
    """Mark session as deleted (hidden from list). File remains on disk. Returns True if updated."""
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["deleted"] = True
        data["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _stream_chat(api_key: str, messages: list[dict], session_id: str = "") -> object:
    """Yields SSE-style events; appends to messages (with images on assistant when applicable)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    max_tool_rounds = 5
    round_count = 0
    current_turn_images: list[str] = []  # persist across tool-call rounds so images are saved

    def _cap_to_recent_images(api_messages: list[dict], max_images: int = MAX_IMAGES_PER_PROMPT) -> list[dict]:
        """Keep at most max_images in the whole conversation, the most recent ones."""
        # Collect (msg_idx, content_idx, part) for every image part in order
        image_slots: list[tuple[int, int, dict]] = []
        for msg_idx, msg in enumerate(api_messages):
            content = msg.get("content")
            if isinstance(content, list):
                for c_idx, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_slots.append((msg_idx, c_idx, part))
        if len(image_slots) <= max_images:
            return api_messages
        # Keep only the last max_images
        keep = set(range(len(image_slots) - max_images, len(image_slots)))
        out = []
        image_idx = 0
        for msg_idx, msg in enumerate(api_messages):
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_parts = []
            for c_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    if image_idx in keep:
                        new_parts.append(part)
                    image_idx += 1
                else:
                    new_parts.append(part)
            out.append({**msg, "content": new_parts})
        return out

    def messages_for_api(msgs: list[dict]) -> list[dict]:
        out = []
        for m in list(msgs):
            role = m.get("role")
            if role == "user":
                content = _user_content_to_api(m.get("content"), session_id)
                out.append({"role": "user", "content": content})
            elif role == "assistant":
                out.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")})
                # So the LLM can discuss how well generated images match intent, inject a user message with the image(s).
                raw_images = m.get("images") or []
                urls = [(x.get("url") if isinstance(x, dict) else x) for x in raw_images if (x.get("url") if isinstance(x, dict) else x)][:MAX_IMAGES_PER_PROMPT]
                if urls:
                    parts = [{"type": "text", "text": "Here are the generated image(s) from your previous turn (so we can discuss how well they match what you intended):"}]
                    for url in urls:
                        data_url = _generated_image_to_data_url(url)
                        if data_url:
                            parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    if len(parts) > 1:
                        out.append({"role": "user", "content": parts})
            else:
                out.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")})
        return _cap_to_recent_images(out)

    while round_count < max_tool_rounds:
        round_count += 1
        payload = {
            "model": CHAT_MODEL,
            "messages": messages_for_api(messages),
            "max_tokens": 16384,
            "temperature": 0.60,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "stream": True,
            "tools": TOOLS,
            "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": True},
        }

        chat_timeout = 120
        try:
            resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=chat_timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            # Server log for diagnosing API errors (e.g. 400 Bad Request, timeouts)
            log.warning(
                "Chat API request failed: %s (request timeout=%ss)",
                e, chat_timeout,
            )
            if hasattr(e, "response") and e.response is not None:
                r = e.response
                log.warning(
                    "Chat API response: status=%s url=%s",
                    r.status_code, r.url,
                )
                try:
                    headers_dict = dict(r.headers)
                    log.warning("Chat API response headers: %s", headers_dict)
                except Exception:
                    pass
                try:
                    body = (r.text or "")[:2000]
                    log.warning("Chat API response body (truncated): %s", body if body else "(empty)")
                except Exception:
                    pass
            yield {"type": "error", "message": str(e)}
            return

        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.strip() == "data: [DONE]":
                break
            if not line_str.startswith("data: "):
                continue
            try:
                j = json.loads(line_str[6:])
            except json.JSONDecodeError:
                continue
            choices = j.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            # Reasoning/thinking (e.g. reasoning_content or reasoning); do not store in message
            reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning_delta:
                yield {"type": "reasoning", "delta": reasoning_delta}

            if delta.get("content"):
                content_parts.append(delta["content"])
                yield {"type": "text", "delta": delta["content"]}

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": tc.get("id") or "",
                        "name": (tc.get("function") or {}).get("name") or "",
                        "arguments": "",
                    }
                fn = tc.get("function") or {}
                tool_calls_acc[idx]["arguments"] += fn.get("arguments") or ""

        full_content = "".join(content_parts)
        tool_calls_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc) if tool_calls_acc[i].get("id") or tool_calls_acc[i].get("name")]

        if not tool_calls_list:
            if full_content or current_turn_images:
                messages.append({"role": "assistant", "content": full_content, "images": current_turn_images})
            yield {"type": "done"}
            return

        assistant_msg = {"role": "assistant", "content": full_content or None, "tool_calls": []}
        for t in tool_calls_list:
            assistant_msg["tool_calls"].append({
                "id": t["id"],
                "type": "function",
                "function": {"name": t["name"], "arguments": t["arguments"]},
            })
        messages.append(assistant_msg)

        for tc in assistant_msg["tool_calls"]:
            name = (tc.get("function") or {}).get("name", "")
            raw_args = (tc.get("function") or {}).get("arguments", "{}")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            result, image_url, img_params = run_tool(name, args, api_key)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })
            if image_url:
                current_turn_images.append({"url": image_url, "params": img_params or {}})
                yield {"type": "image", "url": image_url, "params": img_params or {}}

    yield {"type": "done"}


def create_app():
    from flask import Flask, Response, request, send_from_directory

    app = Flask(__name__, static_folder="static", template_folder="templates")

    @app.route("/")
    def index():
        from flask import render_template
        return render_template("index.html")

    @app.route("/generated_images/<path:filename>")
    def serve_image(filename):
        return send_from_directory(OUTPUT_DIR, filename)

    @app.route("/api/sessions", methods=["GET"])
    def api_sessions():
        return Response(
            json.dumps(load_sessions()),
            mimetype="application/json",
        )

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def api_session_get(session_id):
        session = load_session(session_id)
        if session is None:
            return Response(json.dumps({"error": "Not found"}), status=404, mimetype="application/json")
        # Make generated image URLs absolute so they load correctly on page reload
        base = request.host_url.rstrip("/")
        for m in session.get("messages", []):
            if m.get("role") == "assistant":
                m.setdefault("images", [])
                m["images"] = [
                    (base + u) if (u and isinstance(u, str) and u.startswith("/")) else u
                    for u in m["images"]
                ]
        return Response(json.dumps(session), mimetype="application/json")

    @app.route("/api/sessions/<session_id>", methods=["PATCH"])
    def api_session_patch(session_id):
        data = request.get_json() or {}
        title = (data.get("title") or "").strip()
        if not title:
            return Response(json.dumps({"error": "title required"}), status=400, mimetype="application/json")
        if not update_session_title(session_id, title):
            return Response(json.dumps({"error": "Not found"}), status=404, mimetype="application/json")
        return Response(json.dumps({"id": session_id, "title": title}), mimetype="application/json")

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def api_session_delete(session_id):
        if not soft_delete_session(session_id):
            return Response(json.dumps({"error": "Not found"}), status=404, mimetype="application/json")
        return Response(json.dumps({"id": session_id, "deleted": True}), mimetype="application/json")

    @app.route("/user_images/<path:subpath>")
    def serve_user_image(subpath):
        return send_from_directory(USER_IMAGES_DIR, subpath)

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
        if not api_key:
            return Response(
                json.dumps({"error": "NVIDIA_API_KEY not set. Add it to .env or get one at https://build.nvidia.com/settings/api-keys"}),
                status=503,
                mimetype="application/json",
            )
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        session_id = (data.get("session_id") or "").strip()
        images_data = data.get("images") or []  # list of { "data": "base64...", "mime": "image/jpeg" }
        if not user_message and not images_data:
            return Response(json.dumps({"error": "message or images required"}), status=400, mimetype="application/json")

        if not session_id:
            session_id = str(uuid.uuid4())

        session = load_session(session_id)
        if session is None:
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        else:
            messages = session["messages"]

        # Build user message content: optional images then text (multimodal format for API/storage). API allows at most 8 images per message.
        content_parts = []
        session_images_dir = USER_IMAGES_DIR / session_id
        images_added = 0
        for i, img in enumerate(images_data):
            if images_added >= MAX_IMAGES_PER_PROMPT:
                break
            if not isinstance(img, dict):
                continue
            b64 = (img.get("data") or "").strip()
            if not b64:
                continue
            mime = (img.get("mime") or "image/jpeg").strip().lower()
            ext = "jpg"
            if "png" in mime:
                ext = "png"
            elif "gif" in mime:
                ext = "gif"
            elif "webp" in mime:
                ext = "webp"
            try:
                raw = base64.b64decode(b64)
            except Exception:
                continue
            session_images_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{uuid.uuid4().hex}.{ext}"
            path = session_images_dir / filename
            path.write_bytes(raw)
            url_path = f"/user_images/{session_id}/{filename}"
            content_parts.append({"type": "image_url", "image_url": {"url": url_path}})
            images_added += 1
        if user_message:
            content_parts.append({"type": "text", "text": user_message})
        if not content_parts:
            return Response(json.dumps({"error": "message or images required"}), status=400, mimetype="application/json")
        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            user_content = user_message
        else:
            user_content = content_parts
        messages.append({"role": "user", "content": user_content})

        def generate():
            for event in _stream_chat(api_key, messages, session_id):
                yield f"data: {json.dumps(event)}\n\n"
            title = save_session(session_id, messages)
            yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id, 'title': title})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        print("Set NVIDIA_API_KEY in .env or in the environment. Get a key: https://build.nvidia.com/settings/api-keys")
        raise SystemExit(1)
    app = create_app()
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
