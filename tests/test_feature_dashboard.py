import tempfile
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from analysis.features.visualize.feature_dashboard import (
    _save_prompt_example,
    _safe_path_component,
    make_handler_class,
)


class FeatureDashboardPromptExportTests(unittest.TestCase):
    def test_save_prompt_example_preserves_raw_transcript_in_run_folder(self):
        transcript = (
            "<bos><start_of_turn>user\n"
            "What is 2 + 2?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "The answer is 4"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "prompts"
            data_dir = Path(tmpdir) / "2026.TA.gemma2 2b tc:8192"
            data_dir.mkdir()

            result = _save_prompt_example(
                prompt_output_dir=root,
                data_dir=data_dir,
                payload={
                    "transcript": transcript,
                    "cantor_id": 123,
                    "layer": 1,
                    "feature": 2,
                    "quantile_name": "Top activations (chat)",
                    "example_index": 0,
                },
            )

            saved_path = Path(result["path"])
            self.assertEqual(
                saved_path.parent,
                root / "2026.TA.gemma2_2b_tc_8192",
            )
            self.assertEqual(
                saved_path.name,
                "L1_F2_123_top_activations_chat_01.txt",
            )
            self.assertEqual(saved_path.read_text(), transcript)
            self.assertEqual(result["prompt_format"], "raw")

    def test_save_prompt_example_uses_suffix_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "prompts"
            data_dir = Path(tmpdir) / "run"
            data_dir.mkdir()
            payload = {
                "transcript": "first",
                "cantor_id": 123,
                "layer": 1,
                "feature": 2,
                "quantile_name": "Top",
                "example_index": 0,
            }

            first = _save_prompt_example(root, data_dir, payload)
            second = _save_prompt_example(
                root,
                data_dir,
                {**payload, "transcript": "second"},
            )

            self.assertEqual(Path(first["path"]).name, "L1_F2_123_top_01.txt")
            self.assertEqual(Path(second["path"]).name, "L1_F2_123_top_01_2.txt")
            self.assertEqual(Path(first["path"]).read_text(), "first")
            self.assertEqual(Path(second["path"]).read_text(), "second")

    def test_safe_path_component_rejects_empty_after_sanitizing(self):
        self.assertEqual(_safe_path_component("<<<>>>", fallback="prompt"), "prompt")

    def test_save_prompt_endpoint_writes_raw_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "prompts"
            data_dir = Path(tmpdir) / "run"
            data_dir.mkdir()
            handler = make_handler_class(data_dir, prompt_output_dir=root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            try:
                body = json.dumps({
                    "transcript": "raw transcript",
                    "cantor_id": 7,
                    "layer": 1,
                    "feature": 2,
                    "quantile_name": "Top",
                    "example_index": 0,
                }).encode()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/api/save_prompt",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    payload = json.loads(response.read())
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

            saved_path = Path(payload["path"])
            self.assertEqual(saved_path.read_text(), "raw transcript")
            self.assertEqual(payload["prompt_format"], "raw")


if __name__ == "__main__":
    unittest.main()
