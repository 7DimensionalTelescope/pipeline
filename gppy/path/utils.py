import os
import numpy as np

# def switch_raw_name_order(name):
#     parts = name.split("_")
#     return "_".join(parts[3:5] + parts[0:1] + [format_subseconds_deprecated(parts[6])] + parts[1:3])


def join(*args):
    """broadcasted join"""
    # Convert all inputs to lists (strings become single-element lists)
    lists = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            lists.append(list(arg))
        else:
            lists.append([arg])

    # Determine broadcast length (max list length)
    lengths = [len(lst) for lst in lists]
    max_length = max(lengths) if lengths else 1

    # Validate lengths (1 or max_length)
    for lst in lists:
        if len(lst) not in (1, max_length):
            raise ValueError("All lists must have length 1 or match the longest list")

    # Broadcast lists of length 1 to max_length
    broadcasted = []
    for lst in lists:
        if len(lst) == 1:
            broadcasted.append(lst * max_length)
        else:
            broadcasted.append(lst)

    # Join paths element-wise
    return [os.path.join(*parts) for parts in zip(*broadcasted)]


def broadcast_join(*args):
    """
    Join paths with NumPy-style broadcasting support.
    Handles nested lists/arrays and broadcasts across all dimensions.
    """
    # Convert all inputs to numpy arrays for easier broadcasting
    arrays = []
    for arg in args:
        if isinstance(arg, (str, bytes, os.PathLike)):
            # Convert strings to 0-dimensional arrays
            arrays.append(np.array(arg, dtype=object))
        else:
            # Convert lists/tuples to numpy arrays
            arrays.append(np.array(arg, dtype=object))

    # Use numpy's broadcast mechanism to determine output shape
    try:
        broadcasted = np.broadcast_arrays(*arrays)
    except ValueError as e:
        raise ValueError(f"Cannot broadcast arguments: {e}")

    # Create output array with the broadcasted shape
    output_shape = broadcasted[0].shape
    result = np.empty(output_shape, dtype=object)

    # Iterate through all positions and join paths
    for idx in np.ndindex(output_shape):
        path_components = [arr[idx] for arr in broadcasted]
        result[idx] = os.path.join(*[str(comp) for comp in path_components])

    # Return as list if 1D, otherwise return the numpy array structure
    if result.ndim == 0:
        return str(result.item())
    elif result.ndim == 1:
        return result.tolist()
    else:
        return result.tolist()


# Alternative implementation without numpy dependency
def broadcast_join_pure(*args):
    """
    Pure Python implementation of broadcasting for os.path.join.
    """

    def to_nested_list(arg):
        """Convert input to nested list structure."""
        if isinstance(arg, (str, bytes, os.PathLike)):
            return arg  # Keep strings as scalars
        elif hasattr(arg, "__iter__") and not isinstance(arg, (str, bytes)):
            return [to_nested_list(item) for item in arg]
        else:
            return arg

    def get_shape(obj):
        """Get the shape of a nested list structure."""
        if isinstance(obj, (str, bytes, os.PathLike)) or not hasattr(obj, "__iter__"):
            return ()
        elif len(obj) == 0:
            return (0,)
        else:
            inner_shape = get_shape(obj[0])
            return (len(obj),) + inner_shape

    def broadcast_shapes(*shapes):
        """Determine the broadcasted shape."""
        # Pad shapes to same length
        max_dims = max(len(shape) for shape in shapes) if shapes else 0
        padded_shapes = [(1,) * (max_dims - len(shape)) + shape for shape in shapes]

        # Check compatibility and compute result shape
        result_shape = []
        for dims in zip(*padded_shapes):
            max_dim = max(dims)
            for dim in dims:
                if dim != 1 and dim != max_dim:
                    raise ValueError(f"Cannot broadcast shapes: {shapes}")
            result_shape.append(max_dim)

        return tuple(result_shape)

    def broadcast_to_shape(obj, target_shape):
        """Broadcast object to target shape."""
        obj_shape = get_shape(obj)

        if obj_shape == target_shape:
            return obj

        if obj_shape == ():
            # Scalar case - replicate to fill target shape
            if not target_shape:
                return obj
            return [broadcast_to_shape(obj, target_shape[1:]) for _ in range(target_shape[0])]

        # Handle dimension expansion
        if len(obj_shape) < len(target_shape):
            # Add dimensions at the front
            for _ in range(len(target_shape) - len(obj_shape)):
                obj = [obj]
            return broadcast_to_shape(obj, target_shape)

        # Handle size-1 dimensions
        if obj_shape[0] == 1 and target_shape[0] > 1:
            obj = obj * target_shape[0]

        # Recursively broadcast inner dimensions
        if len(target_shape) > 1:
            obj = [broadcast_to_shape(item, target_shape[1:]) for item in obj]

        return obj

    def join_recursive(components):
        """Recursively join path components."""
        if all(isinstance(comp, (str, bytes, os.PathLike)) or not hasattr(comp, "__iter__") for comp in components):
            return os.path.join(*[str(comp) for comp in components])

        # Find the length of the first dimension
        lengths = [
            len(comp) if hasattr(comp, "__len__") and not isinstance(comp, (str, bytes)) else 1 for comp in components
        ]
        max_length = max(lengths) if lengths else 1

        result = []
        for i in range(max_length):
            inner_components = []
            for comp in components:
                if isinstance(comp, (str, bytes, os.PathLike)) or not hasattr(comp, "__iter__"):
                    inner_components.append(comp)
                else:
                    inner_components.append(comp[i] if len(comp) > 1 else comp[0])
            result.append(join_recursive(inner_components))

        return result

    # Convert inputs and get shapes
    nested_args = [to_nested_list(arg) for arg in args]
    shapes = [get_shape(arg) for arg in nested_args]

    # Determine broadcast shape
    try:
        target_shape = broadcast_shapes(*shapes)
    except ValueError as e:
        raise ValueError(f"Cannot broadcast arguments: {e}")

    # Broadcast all arguments to target shape
    broadcasted_args = [broadcast_to_shape(arg, target_shape) for arg in nested_args]

    # Join paths recursively
    result = join_recursive(broadcasted_args)

    return result


def format_subseconds_deprecated(sec: str):
    """100.0s -> 100s, 0.1s -> 0pt100s"""
    s = float(sec[:-1])
    integer_second = int(s)
    if integer_second != 0:
        return f"{integer_second}s"

    # if subsecond
    millis = int(abs(s) * 1000 + 0.5)  # round to nearest ms
    return f"0pt{millis:03d}s"


def format_subseconds(sec: float) -> str:
    """
    Rules for processed images
    100.0 -> 100, 15.8 -> 15.8
    (Deprecated: 100.0 -> 100, 15.8 -> 15pt8, 0.1 -> 0pt100)
    """
    integer_second = int(sec)
    decimal_second = sec - integer_second
    if decimal_second == 0:
        return f"{integer_second}"
    else:  # if subsecond
        # millis = int(abs(sec) * 1000 + 0.5)  # round to nearest ms
        # return f"0pt{millis:03d}"
        return f"{sec:.1f}"


def strip_binning(binning_string):
    """assume the form 'nbinxnbin'"""
    # pattern = r".*(\dx\d).*"
    # match = re.match(pattern, binning_string)
    # return int(match.group(1)[0])
    return int(binning_string[0])


def format_binning(n_binning: int | str):
    n = str(n_binning)
    return f"{n}x{n}"


def strip_exptime(exptime_string):
    exptime_string.replace("pt", ".")
    return float(exptime_string[:-1])


def format_exptime(exptime: float, type="raw"):
    """type=='raw' is .1f float, others are rounded"""
    if not exptime:  # when the input is not expected to have proper exptime
        return "UndefExptime"  # indicates a bug in a regular pipeline output

    if "raw" in type:
        return f"{exptime:.1f}s"
    else:  # processed
        return format_subseconds(exptime) + "s"


def strip_gain(gain_string):
    """assume the form 'nbinxnbin'"""
    # pattern = r".*(\dx\d).*"
    # match = re.match(pattern, binning_string)
    # return int(match.group(1)[0])
    return int(gain_string[4:])
