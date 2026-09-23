"""Fully decode manifest videos, then optionally remove pairs that reference failures.

Run directly with the training Python environment; no model or accelerator is needed.
Results are checkpointed per video and tied to the manifest and file fingerprints.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import time
from pathlib import Path


def fingerprint(path):
    try:
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns]
    except OSError:
        return None


def alarm_handler(signum, frame):
    raise TimeoutError("Video decode exceeded the configured time limit")


def decode(task):
    import av

    path, relative, required_frame, timeout = task
    before = fingerprint(Path(path))
    row = {"path": relative, "fingerprint": before, "frames": 0, "ok": False}
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)
    av.logging.set_level(av.logging.PANIC)
    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError("No video stream")
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                if frame.is_corrupt:
                    raise ValueError(f"Corrupt decoded frame {row['frames']}")
                # Exercise the same RGB conversion used by training, without retaining frames.
                frame.to_ndarray(format="rgb24")
                row["frames"] += 1
        if row["frames"] == 0 or row["frames"] <= required_frame:
            raise ValueError(f"Only {row['frames']} frames; requires frame {required_frame}")
        if fingerprint(Path(path)) != before:
            raise RuntimeError("Video changed during validation")
        row["ok"] = True
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
    return row


def references(pair):
    for path in pair["robot_videos"]:
        yield path, max(pair["robot_frame_ids"])
    yield pair["human_video"], max(pair["human_frame_ids"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    original = args.manifest.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    manifest = json.loads(original)
    root = Path(manifest["root"])
    required = {}
    for pairs in manifest["splits"].values():
        for pair in pairs:
            for path, frame in references(pair):
                required[path] = max(required.get(path, -1), frame)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.report_dir / "audit.json"
    metadata = {"manifest": str(args.manifest), "sha256": digest, "videos": len(required)}
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Report directory belongs to a different manifest")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    results_path = args.report_dir / "videos.jsonl"
    results = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["path"] in required and row["fingerprint"] == fingerprint(root / row["path"]):
                results[row["path"]] = row
    tasks = [(str(root / p), p, f, args.timeout) for p, f in required.items() if p not in results]
    start = last_print = time.monotonic()
    initial = len(results)
    print(f"Videos={len(required)} cached={initial} pending={len(tasks)} workers={args.workers}", flush=True)
    with results_path.open("a") as output:
        # Ensure an interrupted partial final JSON line cannot swallow a new result.
        output.write("\n")
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=100) as pool:
            for row in pool.imap_unordered(decode, tasks, chunksize=1):
                results[row["path"]] = row
                output.write(json.dumps(row) + "\n")
                output.flush()
                now = time.monotonic()
                if not row["ok"]:
                    print("FAILED " + json.dumps(row), flush=True)
                if now - last_print >= 20:
                    done = len(results)
                    rate = (done - initial) / (now - start)
                    failed = sum(not r["ok"] for r in results.values())
                    print(f"Progress {done}/{len(required)} failed={failed} rate={rate:.2f}/s", flush=True)
                    last_print = now
    assert set(results) == set(required)
    # Retry failures in fresh workers before excluding data on a transient read failure.
    failures = [p for p, row in results.items() if not row["ok"]]
    if failures:
        print(f"Rechecking {len(failures)} failures", flush=True)
        tasks = [(str(root / p), p, required[p], args.timeout * 2) for p in failures]
        with results_path.open("a") as output, mp.get_context("spawn").Pool(args.workers) as pool:
            for row in pool.imap_unordered(decode, tasks, chunksize=1):
                results[row["path"]] = row
                output.write(json.dumps(row) + "\n")
                output.flush()
    bad = {p for p, row in results.items() if not row["ok"]}
    removed = []
    counts = {}
    for split, pairs in manifest["splits"].items():
        kept = []
        for pair in pairs:
            affected = sorted({p for p, _ in references(pair)} & bad)
            if affected:
                removed.append({"split": split, "pair_id": pair["pair_id"], "videos": affected})
            else:
                kept.append(pair)
        counts[split] = {"before": len(pairs), "after": len(kept), "removed": len(pairs) - len(kept)}
        manifest["splits"][split] = kept
    summary = {**metadata, "bad_videos": len(bad), "removed_pairs": len(removed), "splits": counts}
    (args.report_dir / "bad_videos.json").write_text(json.dumps([results[p] for p in sorted(bad)], indent=2) + "\n")
    (args.report_dir / "removed_pairs.json").write_text(json.dumps(removed, indent=2) + "\n")
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.apply:
        if args.manifest.read_bytes() != original:
            raise RuntimeError("Manifest changed during audit; refusing to overwrite")
        for path, row in results.items():
            if row["ok"] and fingerprint(root / path) != row["fingerprint"]:
                raise RuntimeError(f"Video changed since audit: {path}")
        if any(c["before"] and not c["after"] for c in counts.values()):
            raise RuntimeError("Refusing to empty a split")
        backup = args.report_dir / "manifest.original.json"
        if backup.exists():
            if backup.read_bytes() != original:
                raise RuntimeError("Existing backup differs")
        else:
            shutil.copy2(args.manifest, backup)
        temporary = args.manifest.with_name(args.manifest.name + ".audit.tmp")
        with temporary.open("x") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        assert json.loads(temporary.read_text()) == manifest
        shutil.copymode(args.manifest, temporary)
        os.replace(temporary, args.manifest)
        summary["applied"] = True
        summary["backup"] = str(backup)
        summary["updated_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
