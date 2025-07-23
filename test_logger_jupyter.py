#!/usr/bin/env python3
"""
Test script to verify the updated Logger works correctly in Jupyter environments.

This script simulates the Jupyter notebook behavior and tests that:
1. Logger without redirection doesn't affect global stdout/stderr
2. Logger with redirection works as expected
3. Context manager properly restores streams
"""

import sys
import io
from gppy.services.logger import Logger

def test_logger_without_redirection():
    """Test that logger without redirection doesn't affect global streams."""
    print("=== Testing Logger without redirection ===")
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create logger without redirection
    logger = Logger(
        name="Test Logger",
        log_file="test_no_redirect.log",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    
    # Test that stdout/stderr are unchanged
    assert sys.stdout is original_stdout, "stdout should be unchanged"
    assert sys.stderr is original_stderr, "stderr should be unchanged"
    
    # Test logging
    logger.info("This is a test message")
    logger.error("This is a test error")
    
    # Test that print still goes to original stdout
    print("This should appear in the original stdout")
    
    print("✓ Logger without redirection works correctly")
    return logger

def test_logger_with_redirection():
    """Test that logger with redirection works as expected."""
    print("=== Testing Logger with redirection ===")
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create logger with redirection
    logger = Logger(
        name="Test Logger",
        log_file="test_with_redirect.log",
        redirect_stdout=True,
        redirect_stderr=True,
    )
    
    # Test that stdout/stderr are redirected
    assert sys.stdout is not original_stdout, "stdout should be redirected"
    assert sys.stderr is not original_stderr, "stderr should be redirected"
    
    # Test logging
    logger.info("This is a test message with redirection")
    logger.error("This is a test error with redirection")
    
    # Test that print goes to logger
    print("This should go to the logger")
    
    # Restore original streams
    logger.restore_stdout_stderr()
    
    # Test that streams are restored
    assert sys.stdout is original_stdout, "stdout should be restored"
    assert sys.stderr is original_stderr, "stderr should be restored"
    
    print("✓ Logger with redirection works correctly")
    return logger

def test_context_manager():
    """Test that context manager properly handles redirection."""
    print("=== Testing Logger context manager ===")
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create logger without initial redirection
    logger = Logger(
        name="Test Logger",
        log_file="test_context.log",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    
    # Test that streams are unchanged initially
    assert sys.stdout is original_stdout, "stdout should be unchanged initially"
    assert sys.stderr is original_stderr, "stderr should be unchanged initially"
    
    # Test context manager
    with logger:
        # Manually redirect within context
        logger.redirect_stdout_stderr()
        
        # Test that streams are redirected
        assert sys.stdout is not original_stdout, "stdout should be redirected in context"
        assert sys.stderr is not original_stderr, "stderr should be redirected in context"
        
        # Test logging
        logger.info("This is a test message in context")
        print("This should go to the logger in context")
    
    # Test that streams are restored after context
    assert sys.stdout is original_stdout, "stdout should be restored after context"
    assert sys.stderr is original_stderr, "stderr should be restored after context"
    
    print("✓ Logger context manager works correctly")

def test_backward_compatibility():
    """Test that existing code continues to work."""
    print("=== Testing backward compatibility ===")
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create logger with old-style constructor (no redirection parameters)
    logger = Logger(
        name="Backward Compatible Logger",
        log_file="test_backward.log",
    )
    
    # Test that streams are unchanged (since redirection defaults to False)
    assert sys.stdout is original_stdout, "stdout should be unchanged for backward compatibility"
    assert sys.stderr is original_stderr, "stderr should be unchanged for backward compatibility"
    
    # Test logging
    logger.info("This is a backward compatibility test")
    logger.error("This is a backward compatibility error test")
    
    print("✓ Backward compatibility works correctly")
    return logger

def test_jupyter_simulation():
    """Simulate Jupyter notebook behavior."""
    print("=== Simulating Jupyter notebook behavior ===")
    
    # Simulate cell 1: Setup logger
    print("Cell 1: Setting up logger...")
    logger = Logger(
        name="Jupyter Logger",
        log_file="jupyter_test.log",
        redirect_stdout=False,  # Important for Jupyter
        redirect_stderr=False,
    )
    logger.info("Logger setup complete")
    
    # Simulate cell 2: Use logger
    print("Cell 2: Using logger...")
    logger.info("Processing started")
    # Simulate some processing
    print("Processing step 1")
    print("Processing step 2")
    logger.info("Processing completed")
    
    # Simulate cell 3: Check results
    print("Cell 3: Checking results...")
    print("Results should appear in this cell")
    logger.info("Additional logging in cell 3")
    
    print("✓ Jupyter simulation works correctly")
    return logger

if __name__ == "__main__":
    print("Running Logger tests for Jupyter compatibility...")
    
    try:
        # Run all tests
        test_logger_without_redirection()
        test_logger_with_redirection()
        test_context_manager()
        test_backward_compatibility()
        test_jupyter_simulation()
        
        print("\n🎉 All tests passed! The Logger is now Jupyter-compatible.")
        print("\nUsage in Jupyter notebooks:")
        print("1. Create logger with redirect_stdout=False, redirect_stderr=False")
        print("2. Use logger.info(), logger.error(), etc. for logging")
        print("3. Use print() for output that should appear in the current cell")
        print("4. Use context manager for temporary redirection if needed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 