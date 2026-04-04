from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from local_model_chat.backends import (
    Backend,
    BackendFactorySettings,
    BackendManager,
    CompletionMetrics,
    CompletionResult,
)
from local_model_chat.presets import get_preset, resolve_initial_preset_id
from local_model_chat.server import AppServer


class FakeBackend(Backend):
    runtime_name = "Fake"

    def __init__(self, preset_id: str):
        self.display_model = preset_id
        self.closed = False

    def generate(self, messages, max_tokens):
        if messages and messages[0]["role"] == "system" and "compress a conversation" in messages[0]["content"].lower():
            return CompletionResult(
                text="user wants to keep testing local model presets",
                metrics=CompletionMetrics(generation_tokens=11),
            )
        last_user = ""
        for message in reversed(messages):
            if message["role"] == "user":
                last_user = message["content"]
                break
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
            "gemma4-31b-llama",
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

            request = Request(
                f"{base_url}/api/chat",
                data=json.dumps(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "conversation_summary": "prior summary",
                        "max_tokens": 64,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertIn("hello", payload["reply"])

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
