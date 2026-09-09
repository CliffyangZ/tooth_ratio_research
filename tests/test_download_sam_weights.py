import hashlib
import tempfile
import unittest
from pathlib import Path

from src.models.download_sam_weights import download_checkpoint


class DownloadCheckpointTests(unittest.TestCase):
    def test_downloads_checkpoint_and_verifies_sha256(self) -> None:
        payload = b"test SAM checkpoint bytes"
        expected_sha256 = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.pth"
            destination = temp_path / "weights" / "model.pth"
            source.write_bytes(payload)

            downloaded = download_checkpoint(
                source.as_uri(), destination, expected_sha256
            )

            self.assertTrue(downloaded)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".pth.part").exists())


if __name__ == "__main__":
    unittest.main()
