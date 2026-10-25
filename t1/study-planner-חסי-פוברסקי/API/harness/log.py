"""Every runtime action is printed with the name of the harness file that performed it."""
import sys

QUIET = False


def log(source: str, msg: str) -> None:
    if not QUIET:
        print(f"[{source:<14}] {msg}", file=sys.stderr, flush=True)
