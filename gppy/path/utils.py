import re


def switch_raw_name_order(name):
    parts = name.split("_")
    return "_".join(parts[3:5] + parts[0:1] + [format_subseconds_deprecated(parts[6])] + parts[1:3])


def format_subseconds_deprecated(sec: str):
    """100.0s -> 100s, 0.1s -> 0pt100s"""
    s = float(sec[:-1])
    integer_second = int(s)
    if integer_second != 0:
        return f"{integer_second}s"

    # if subsecond
    millis = int(abs(s) * 1000 + 0.5)  # round to nearest ms
    return f"0pt{millis:03d}s"


def format_subseconds(sec: float):
    """100.0 -> 100, 0.1 -> 0pt100"""
    integer_second = int(sec)
    if integer_second != 0:
        return f"{integer_second}"
    else:  # if subsecond
        millis = int(abs(sec) * 1000 + 0.5)  # round to nearest ms
        return f"0pt{millis:03d}"


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
    return float(exptime_string[:-1])


def format_exptime(exptime: float, type="raw_image"):
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
