"""Download the official Segment Anything ViT-B checkpoint.

Usage:
    python src/models/download_sam_weights.py

The checkpoint is stored at ``src/models/weights/sam_vit_b_01ec64.pth`` by
default. Existing files are reused only when their SHA-256 digest matches the
official checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

SAM_VIT_B_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
SAM_VIT_B_SHA256 = (
    "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
)
DEFAULT_DESTINATION = Path("src/models/weights/sam_vit_b_01ec64.pth")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    url: str,
    destination: Path,
    expected_sha256: str,
    chunk_size: int = 1024 * 1024,
) -> bool:
    """Download and verify a checkpoint atomically.

    Returns ``True`` when a file was downloaded and ``False`` when an existing,
    checksum-valid checkpoint was reused.
    """
    destination = Path(destination)
    expected_sha256 = expected_sha256.lower()

    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    temporary_path.unlink(missing_ok=True)

    try:
        with urllib.request.urlopen(url) as response, temporary_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(chunk_size):
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(
                        f"\rDownloading: {percent:6.2f}% "
                        f"({downloaded}/{total} bytes)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

        if total:
            print(file=sys.stderr)

        actual_sha256 = sha256_file(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        os.replace(temporary_path, destination)
        return True
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SAM_VIT_B_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--sha256", default=SAM_VIT_B_SHA256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloaded = download_checkpoint(args.url, args.output, args.sha256)
    action = "Downloaded" if downloaded else "Already verified"
    print(f"{action}: {args.output}")
    print(f"SHA-256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
