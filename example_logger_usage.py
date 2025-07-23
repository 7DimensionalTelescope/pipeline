#!/usr/bin/env python3
"""
Example usage of the updated Logger class for Jupyter notebooks.

This script demonstrates how to use the Logger without global stdout/stderr redirection,
which prevents the issue where logs appear in the wrong Jupyter cell.
"""

from gppy.services.logger import Logger

# Example 1: Basic usage without redirection (recommended for Jupyter)
def example_basic_usage():
    """Create a logger without redirecting stdout/stderr."""
    logger = Logger(
        name="My Pipeline Logger",
        log_file="pipeline.log",
        level="INFO",
        redirect_stdout=False,  # Don't redirect stdout
        redirect_stderr=False,  # Don't redirect stderr
    )
    
    logger.info("This message will only go to the log file and console handlers")
    logger.error("This error will only go to the log file and console handlers")
    
    # Regular print statements will go to the original stdout (Jupyter cell output)
    print("This print statement goes to the original stdout")
    
    return logger

# Example 2: Using context manager for temporary redirection
def example_context_manager():
    """Use logger as a context manager for temporary redirection."""
    logger = Logger(
        name="Context Logger",
        log_file="context_pipeline.log",
        level="DEBUG",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    
    # Outside the context, prints go to original stdout
    print("Before context: This goes to original stdout")
    
    with logger:
        # Inside the context, stdout/stderr are redirected
        logger.redirect_stdout_stderr()
        print("Inside context: This goes to the logger")
        logger.info("Logger message inside context")
    
    # After context, stdout is restored
    print("After context: This goes back to original stdout")
    
    return logger

# Example 3: Manual redirection control
def example_manual_control():
    """Manually control stdout/stderr redirection."""
    logger = Logger(
        name="Manual Control Logger",
        log_file="manual_pipeline.log",
        level="INFO",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    
    print("Before redirection: Original stdout")
    
    # Manually redirect
    logger.redirect_stdout_stderr()
    print("After redirection: This goes to logger")
    logger.info("Logger message after manual redirection")
    
    # Manually restore
    logger.restore_stdout_stderr()
    print("After restoration: Back to original stdout")
    
    return logger

# Example 4: Jupyter notebook usage pattern
def jupyter_notebook_example():
    """
    Recommended usage pattern for Jupyter notebooks:
    
    # Cell 1: Setup logger
    from gppy.services.logger import Logger
    logger = Logger(
        name="Pipeline Logger",
        log_file="pipeline.log",
        redirect_stdout=False,  # Important: don't redirect in Jupyter
        redirect_stderr=False,
    )
    
    # Cell 2: Use logger
    logger.info("Processing started")
    # Your pipeline code here
    logger.info("Processing completed")
    
    # Cell 3: Check results
    print("Results will appear in this cell")
    logger.info("Additional logging")
    """
    logger = Logger(
        name="Jupyter Logger",
        log_file="jupyter_pipeline.log",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    
    logger.info("Jupyter-compatible logger setup complete")
    print("This print statement appears in the current cell")
    
    return logger

if __name__ == "__main__":
    print("Running logger examples...")
    
    print("\n=== Example 1: Basic usage ===")
    logger1 = example_basic_usage()
    
    print("\n=== Example 2: Context manager ===")
    logger2 = example_context_manager()
    
    print("\n=== Example 3: Manual control ===")
    logger3 = example_manual_control()
    
    print("\n=== Example 4: Jupyter pattern ===")
    logger4 = jupyter_notebook_example()
    
    print("\nAll examples completed successfully!") 