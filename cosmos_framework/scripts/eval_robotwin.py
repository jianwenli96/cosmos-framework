# SPDX-License-Identifier: OpenMDW-1.1
"""Run RoboTwin expert-filtered closed-loop episodes against a Cosmos HTTP server."""

import argparse

from cosmos_framework.evaluation.robotwin.runner import EvaluationConfigError, run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-seed", type=int)
    parser.add_argument("--execute-steps", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=float, default=10)
    parser.add_argument("--video-stride", type=int, default=5)
    parser.add_argument("--skip-render-check", action="store_true", help="Explicitly use RoboTwin renderer defaults")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        run_evaluation(args)
    except EvaluationConfigError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
