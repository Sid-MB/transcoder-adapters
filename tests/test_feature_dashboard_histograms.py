import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from analysis.features.activation_histograms import save_activation_histograms_npz
from analysis.features.visualize.feature_dashboard import (
    _load_feature_histogram_payload,
    make_handler_class,
)


class FeatureDashboardHistogramTests(unittest.TestCase):
    def test_load_feature_histogram_payload_returns_small_json_safe_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            bins = np.array([0.0, 1.0, 2.0], dtype=np.float32)
            save_activation_histograms_npz(
                path=data_dir / "activation_histograms.npz",
                bin_lower_bounds=bins,
                domain_names=["chat", "fineweb"],
                run_hist_total=np.array([10, 20, 30], dtype=np.uint64),
                run_hist_by_domain=np.array([[8, 9, 10], [2, 11, 20]], dtype=np.uint64),
                feature_hist_total_by_layer=[
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
                ],
                feature_hist_by_domain_by_layer=[
                    np.array(
                        [
                            [[1, 0, 0], [2, 0, 0]],
                            [[0, 2, 3], [0, 5, 6]],
                        ],
                        dtype=np.uint32,
                    )
                ],
            )

            payload = _load_feature_histogram_payload(
                data_dir=data_dir,
                histograms_file="activation_histograms.npz",
                feature_meta={"layer": 0, "feature": 1},
                tokens_per_domain={"chat": 100, "fineweb": 200},
                layer_cache={},
            )
            self.assertEqual(payload["bin_lower_bounds"], [0.0, 1.0, 2.0])
            self.assertEqual(payload["domain_names"], ["chat", "fineweb"])
            self.assertEqual(payload["total_counts"], [4, 5, 6])
            self.assertEqual(payload["by_domain_counts"]["chat"], [2, 0, 0])
            self.assertEqual(payload["by_domain_counts"]["fineweb"], [0, 5, 6])
            self.assertEqual(payload["by_domain_token_density"]["chat"], [0.02, 0.0, 0.0])
            self.assertAlmostEqual(
                payload["by_domain_activation_fraction"]["fineweb"][1],
                5 / 11,
            )
            json.dumps(payload)

    def test_feature_histogram_route_returns_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "features").mkdir()
            (data_dir / "feature_metadata.json").write_text(json.dumps({
                "activation_histograms_file": "activation_histograms.npz",
                "tokens_per_domain": {"chat": 100, "fineweb": 200},
                "features": [{"cantor_id": 2, "layer": 0, "feature": 1}],
            }))
            save_activation_histograms_npz(
                path=data_dir / "activation_histograms.npz",
                bin_lower_bounds=np.array([0.0, 1.0, 2.0], dtype=np.float32),
                domain_names=["chat", "fineweb"],
                run_hist_total=np.array([10, 20, 30], dtype=np.uint64),
                run_hist_by_domain=np.array([[8, 9, 10], [2, 11, 20]], dtype=np.uint64),
                feature_hist_total_by_layer=[
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
                ],
                feature_hist_by_domain_by_layer=[
                    np.array(
                        [
                            [[1, 0, 0], [2, 0, 0]],
                            [[0, 2, 3], [0, 5, 6]],
                        ],
                        dtype=np.uint32,
                    )
                ],
            )

            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler_class(data_dir))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/feature_hist/2",
                    timeout=5,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["total_counts"], [4, 5, 6])
                self.assertIn("by_domain_token_density", payload)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
