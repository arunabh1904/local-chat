from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from local_model_chat.backends import (
    Backend,
    BackendFactorySettings,
    BackendManager,
    CompletionMetrics,
    CompletionResult,
)
from local_model_chat.model_cache import configure_model_cache, default_model_cache_dir
from local_model_chat.presets import get_preset, resolve_initial_preset_id
from local_model_chat.server import AppServer


TEST_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9p2L1ZkAAAAASUVORK5CYII="
)


def flatten_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                pieces.append(item.get("text", ""))
            elif item.get("type") == "image":
                pieces.append("[image]")
        return " ".join(piece for piece in pieces if piece).strip()
    return ""


class FakeBackend(Backend):
    runtime_name = "Fake"

    def __init__(self, preset_id: str):
        self.display_model = preset_id
        self.closed = False

    def generate(self, messages, max_tokens, conversation_image=None):
        if (
            messages
            and messages[0]["role"] == "system"
            and "compress a conversation" in flatten_content(messages[0]["content"]).lower()
        ):
            return CompletionResult(
                text="user wants to keep testing local model presets",
                metrics=CompletionMetrics(generation_tokens=11),
            )
        last_user = ""
        for message in reversed(messages):
            if message["role"] == "user":
                last_user = flatten_content(message["content"])
                break
        if conversation_image is not None:
            last_user = f"{last_user} [image:{conversation_image.name}]".strip()
        return CompletionResult(
            text=f"{self.display_model}:{last_user}",
            metrics=CompletionMetrics(prompt_tokens=42, generation_tokens=8),
        )

    def close(self):
        self.closed = True


def fake_backend_factory(preset, settings):
    del settings
    return FakeBackend(preset.id)


class LocalModelChatTests(unittest.TestCase):
    def test_legacy_gemma_aliases_map_to_presets(self):
        self.assertEqual(
            resolve_initial_preset_id(None, "mlx", "26b"),
            "gemma4-26b-a4b-mlx",
        )
        self.assertEqual(
            resolve_initial_preset_id(None, "llama", "31b"),
            "gemma4-31b-mlx",
        )

    def test_backend_manager_switches_and_summarizes(self):
        manager = BackendManager(
            initial_preset=get_preset("gemma4-e2b-mlx"),
            settings=BackendFactorySettings(),
            backend_factory=fake_backend_factory,
        )
        summary = manager.switch_preset(
            preset_id="qwen3-14b-mlx",
            messages=[{"role": "user", "content": "continue the benchmark"}],
            conversation_summary="",
        )
        self.assertIn("keep testing", summary)
        self.assertEqual(manager.snapshot()["current_preset"]["id"], "qwen3-14b-mlx")
        manager.close()

    def test_backend_manager_can_load_in_background_with_cache_notice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BackendManager(
                initial_preset=get_preset("gemma4-e2b-mlx"),
                settings=BackendFactorySettings(model_cache_dir=temp_dir),
                backend_factory=fake_backend_factory,
                autoload=False,
            )
            snapshot = manager.snapshot()
            self.assertEqual(snapshot["state"], "loading")
            self.assertIn("Downloading Gemma 4 E2B", snapshot["cache_notice"])
            self.assertIn(str(Path(temp_dir) / "huggingface" / "hub"), snapshot["cache_notice"])

            manager.start_loading()
            for _ in range(20):
                if manager.snapshot()["state"] == "ready":
                    break
                time.sleep(0.01)
            self.assertEqual(manager.snapshot()["state"], "ready")
            self.assertEqual(manager.snapshot()["cache_notice"], "")
            manager.close()

    def test_configure_model_cache_sets_hf_environment(self):
        tracked = [
            "LOCAL_CHAT_MODEL_CACHE_DIR",
            "HF_HOME",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
        ]
        previous = {name: os.environ.get(name) for name in tracked}
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = configure_model_cache(temp_dir)
                self.assertEqual(cache_dir, Path(temp_dir).resolve())
                self.assertEqual(os.environ["LOCAL_CHAT_MODEL_CACHE_DIR"], str(cache_dir))
                self.assertEqual(os.environ["HF_HOME"], str(cache_dir / "huggingface"))
                self.assertEqual(
                    os.environ["HF_HUB_CACHE"],
                    str(cache_dir / "huggingface" / "hub"),
                )
                self.assertEqual(
                    os.environ["TRANSFORMERS_CACHE"],
                    str(cache_dir / "huggingface" / "transformers"),
                )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_default_model_cache_dir_uses_home_cache(self):
        self.assertEqual(default_model_cache_dir(), Path.home() / ".cache")

    def test_http_endpoints(self):
        manager = BackendManager(
            initial_preset=get_preset("gemma4-e2b-mlx"),
            settings=BackendFactorySettings(),
            backend_factory=fake_backend_factory,
        )
        server = AppServer(("127.0.0.1", 0), manager, default_max_tokens=512)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"

            with urlopen(f"{base_url}/api/info") as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["current_preset"]["id"], "gemma4-e2b-mlx")
            self.assertGreater(len(payload["presets"]), 1)
            self.assertIn("model_cache_dir", payload)
            self.assertTrue(payload["image_chat_available"])
            self.assertEqual(payload["vision_preset"]["id"], "qwen35-35b-a3b-mlx")
            self.assertFalse(payload["current_preset"]["supports_images"])
            self.assertTrue(any(item["supports_images"] for item in payload["presets"]))

            request = Request(
                f"{base_url}/api/chat",
                data=json.dumps(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "conversation_summary": "prior summary",
                        "max_tokens": 64,
                        "conversation_image": {
                            "name": "tiny.png",
                            "media_type": "image/png",
                            "data_url": TEST_IMAGE,
                        },
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertIn("hello", payload["reply"])
            self.assertIn("tiny.png", payload["reply"])
            self.assertEqual(payload["current_preset"]["id"], "qwen35-35b-a3b-mlx")

            mismatched_image_request = Request(
                f"{base_url}/api/chat",
                data=json.dumps(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "conversation_image": {
                            "name": "tiny.png",
                            "media_type": "image/png",
                            "data_url": TEST_IMAGE.replace("data:image/png", "data:image/jpeg"),
                        },
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as mismatched_image_error:
                urlopen(mismatched_image_request)
            self.assertEqual(mismatched_image_error.exception.code, 400)

            switch_request = Request(
                f"{base_url}/api/switch-preset",
                data=json.dumps(
                    {
                        "preset_id": "qwen3-14b-mlx",
                        "messages": [{"role": "user", "content": "hello"}],
                        "conversation_summary": "",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(switch_request) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["current_preset"]["id"], "qwen3-14b-mlx")
            self.assertIn("testing local model presets", payload["conversation_summary"])

            bad_request = Request(
                f"{base_url}/api/chat",
                data=json.dumps(
                    {
                        "messages": [{"role": "system", "content": "bad"}],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError):
                urlopen(bad_request)
        finally:
            server.shutdown()
            server.server_close()
            manager.close()


if __name__ == "__main__":
    unittest.main()
