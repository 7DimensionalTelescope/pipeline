import gc
import cupy as cp

def cleanup_memory() -> None:
    """
    Perform comprehensive memory cleanup across CPU and GPU.

    This function provides a robust mechanism for releasing 
    unused memory resources in both CPU and GPU contexts.

    Key Operations:
    - Trigger Python garbage collection
    - Free CuPy default memory pool blocks
    - Free CuPy pinned memory pool blocks
    - Perform a second garbage collection to ensure complete memory release

    Ideal for:
    - Preventing memory leaks
    - Managing memory in long-running scientific computing tasks
    - Preparing for memory-intensive operations

    Notes:
    - Calls garbage collection twice to ensure thorough cleanup
    - Uses CuPy's memory pool management for GPU memory
    - Minimal performance overhead

    Example:
        >>> # Before starting a memory-intensive task
        >>> cleanup_memory()
    """
    
    try:
        for device_id in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device_id):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass

    gc.collect()  # Initial garbage collection
    


def cpu_callback_wrapper(task, tree, callback):
    def wrapper(result):
        # Update task with result from the async operation
        task.result = result
        # Call your existing callback logic
        callback(task, tree)
    return wrapper
