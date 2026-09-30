# SPDX-License-Identifier: OpenMDW-1.1
"""Standard-library-only health checks and result summaries for remote evaluation."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

TASKS = Path(__file__).with_name("tasks.txt").read_text().splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["wait", "summary"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--execute-steps", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if args.command == "summary":
        rows = []
        for task in TASKS:
            path = output / task / "metrics.json"
            result = json.loads(path.read_text()) if path.is_file() else {}
            rows.append(
                dict(
                    task=task,
                    completed=result.get("completed", 0),
                    successes=result.get("successes", 0),
                    success_rate=result.get("success_rate"),
                )
            )
        (output / "summary.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows, indent=2))
        return
    deadline = time.monotonic() + args.timeout
    pending = dict(enumerate(TASKS))
    infos = {}
    opener = build_opener(ProxyHandler({}))
    while pending:
        for index, task in list(pending.items()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Services not ready: {list(pending.values())}")
            url = f"http://{args.host}:{args.port + index}/info"
            try:
                with opener.open(url, timeout=min(2, remaining)) as response:
                    info = json.load(response)
            except (URLError, TimeoutError, ConnectionError):
                continue
            if info.get("task") != task:
                raise ValueError(f"{url}: expected task {task}, received {info.get('task')}")
            if not info.get("supports_episode_instruction"):
                raise ValueError(f"{url}: server does not support episode instructions; upgrade it")
            if info.get("action_horizon", 0) < args.execute_steps:
                raise ValueError(f"{task}: execute_steps exceeds server action_horizon")
            infos[task] = dict(url=url, **info)
            del pending[index]
            print(f"Ready: {task} at {url}", flush=True)
        if pending:
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    (output / "services.json").write_text(json.dumps(infos, indent=2))


if __name__ == "__main__":
    main()
