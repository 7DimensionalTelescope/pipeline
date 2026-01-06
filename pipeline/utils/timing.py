import time
from datetime import datetime


def lapse(explanation="elapsed", print_output=True, reset=False):
    """
    Measure and report elapsed time using a global checkpoint.

    A utility function for performance tracking and logging elapsed time
    between function calls. It supports various time unit representations
    and optional console output.

    Args:
        explanation (str, optional): Description for the elapsed time report.
            Defaults to "elapsed".
        print_output (bool, optional): Whether to print the time report.
            Defaults to True.

    Returns:
        float: Elapsed time in seconds

    Usage:
        >>> lapse("Start")  # Initializes the timer
        >>> # Do some work
        >>> lapse("Task completed")  # Prints elapsed time
    """
    from timeit import default_timer as timer

    global _dhutil_lapse_checkpoint  # Global Checkpoint
    # Initialize if not yet defined
    if "_dhutil_lapse_checkpoint" not in globals():
        _dhutil_lapse_checkpoint = None

    current_time = timer()

    # Initialize if it's the first call
    if reset or _dhutil_lapse_checkpoint is None:
        _dhutil_lapse_checkpoint = current_time
        if print_output:
            print(f"Timer (re)started", end="\n")
        return 0.0

    elapsed_time = current_time - _dhutil_lapse_checkpoint

    if elapsed_time < 60:
        dt, unit = elapsed_time, "seconds"
    elif elapsed_time > 3600:
        dt, unit = elapsed_time / 3600, "hours"
    else:
        dt, unit = elapsed_time / 60, "minutes"

    _dhutil_lapse_checkpoint = current_time  # Update the checkpoint

    print_str = f"{dt:.3f} {unit} {explanation}"
    # print(print_str)  # log the elapsed time at INFO level

    if print_output:
        print(print_str, end="\n")  # log the elapsed time
    return elapsed_time  # in seconds


def time_diff_in_seconds(datetime1, datetime2=None, return_float=False):
    if datetime2 is None:
        datetime2 = time.time()
    if isinstance(datetime1, datetime):
        datetime1 = datetime1.timestamp()
    if isinstance(datetime2, datetime):
        datetime2 = datetime2.timestamp()
    time_diff = datetime2 - datetime1

    if return_float:
        return abs(time_diff)
    else:
        return f"{abs(time_diff):.2f}"
