import select
import sys

from ..const import IS_PIPELINE_LOCK, PIPELINE_LOCK_WAIT_SECONDS


def enforce_pipeline_lock(*, action, requested_is_pipeline=None, wait_seconds=PIPELINE_LOCK_WAIT_SECONDS):
    """
    Requires explicit consent when running with is_pipeline=False (runtime) and IS_PIPELINE=True (environment variable)
    """
    if not IS_PIPELINE_LOCK:
        return

    if requested_is_pipeline is not False:
        return

    _require_explicit_consent(action=action, wait_seconds=wait_seconds)


def _require_explicit_consent(*, action, wait_seconds):
    print(f"[WARNING] {action}")
    print("[WARNING] IS_PIPELINE=True, but this run requested is_pipeline=False.")
    print("[WARNING] Explicit confirmation is required before continuing.")

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Explicit consent is required when IS_PIPELINE=True and is_pipeline=False, "
            "but no interactive terminal is available."
        )

    print(f"Type YES within {wait_seconds}s to continue, or press Ctrl+C to cancel: ", end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], wait_seconds)
    print()

    if not ready:
        raise RuntimeError(f"Timed out waiting for consent after {wait_seconds}s.")

    if sys.stdin.readline().strip() != "YES":
        raise RuntimeError("Pipeline lock cancelled the run because explicit consent was not provided.")
