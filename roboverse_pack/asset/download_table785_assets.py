#!/usr/bin/env python3
"""Download all assets for table785 config."""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
UUIDS = [
    "126f60baf12759ea957fb6c38ba7458d",
    "1522dad65f0859758dad5636ba348bf8",
    "18848428c54456aa82070f2fd33f7bb4",
    "b4b40966ebda5393bd4d7fc634062519",
    "848396479c0b5da3bc05d0ef74d4dcfb",
]


def main():
    script = project_root / "generation" / "download_asset.py"
    target = "dataset/basic_furniture/table"

    print(f"Downloading {len(UUIDS)} table assets...\n")

    failed = []
    for i, uuid in enumerate(UUIDS, 1):
        print(f"[{i}/{len(UUIDS)}] {uuid}")
        cmd = [sys.executable, str(script), "--target_type", target, "--uuid", uuid]

        try:
            subprocess.run(cmd, check=True, cwd=str(project_root))
        except subprocess.CalledProcessError:
            failed.append(uuid)

    print("\n" + "=" * 50)
    if failed:
        print(f"Failed: {len(failed)}/{len(UUIDS)}")
        for uuid in failed:
            print(f"  - {uuid}")
        sys.exit(1)
    else:
        print(f"Success: All {len(UUIDS)} assets downloaded")


if __name__ == "__main__":
    main()
