import os
import psutil
import resource


def get_fd_info():
    """
    Get current file descriptor usage and limit information.

    Returns:
        dict: Dictionary with keys:
            - current: Current number of open file descriptors
            - soft_limit: Soft limit for file descriptors
            - hard_limit: Hard limit for file descriptors
            - percent_used: Percentage of soft limit used
            - pid: Process ID
    """
    try:
        # Get current FD count by listing /proc/self/fd
        fd_count = len(os.listdir("/proc/self/fd"))
    except (OSError, FileNotFoundError):
        # Fallback for systems without /proc (e.g., macOS, Windows)
        try:
            process = psutil.Process()
            fd_count = len(process.open_files())
        except:
            fd_count = -1

    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, AttributeError):
        soft_limit = hard_limit = -1

    percent_used = (fd_count / soft_limit * 100) if soft_limit > 0 else -1

    return {
        "current": fd_count,
        "soft_limit": soft_limit,
        "hard_limit": hard_limit,
        "percent_used": percent_used,
        "pid": os.getpid(),
    }


def log_fd_info(logger=None, prefix=""):
    """
    Log file descriptor information to logger or print to console.

    Args:
        logger: Logger instance (optional). If None, prints to console.
        prefix: Optional prefix string for the log message.
    """
    info = get_fd_info()
    msg = f"{prefix}FD usage: {info['current']}/{info['soft_limit']} ({info['percent_used']:.1f}%) [PID: {info['pid']}]"

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
