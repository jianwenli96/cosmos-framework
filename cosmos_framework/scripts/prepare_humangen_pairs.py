# SPDX-License-Identifier: OpenMDW-1.1
"""Create a read-only HumanGen index outside the original dataset."""

import argparse
import json

from cosmos_framework.data.generator.action.datasets.humangen_dataset import (
    SOURCES,
    build_humangen_manifest,
    safe_output,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    output = safe_output(args.root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {output}")
    payload = build_humangen_manifest(
        args.root,
        args.sources.split(","),
        args.seed,
        args.val_ratio,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication, with no overwrite even if two cluster ranks prepare concurrently.
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        temporary = stream.name
        json.dump(payload, stream, indent=2)
    try:
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    print(
        json.dumps(
            dict(
                output=str(output),
                counts={k: len(v) for k, v in payload["splits"].items()},
                rejected=payload["rejected"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
