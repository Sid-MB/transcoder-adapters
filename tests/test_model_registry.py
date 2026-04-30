import unittest

import models


class ModelRegistryTests(unittest.TestCase):
    def tearDown(self):
        models._REGISTRY.clear()

    def test_detect_gemma4_does_not_import_backend(self):
        self.assertEqual(models.detect_architecture("google/gemma-4-E2B-it"), "gemma4")
        self.assertNotIn("gemma4", models._REGISTRY)

    def test_unknown_architecture_lists_static_architectures(self):
        with self.assertRaisesRegex(ValueError, "gemma4"):
            models.get_transcoder_classes("not-real")


if __name__ == "__main__":
    unittest.main()
