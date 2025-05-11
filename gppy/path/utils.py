import re


def switch_raw_name_order(name):
    parts = name.split("_")
    return "_".join(parts[3:5] + parts[0:1] + [format_subseconds(parts[6])] + parts[1:3])


def format_subseconds(sec: str):
    """100.0s -> 100s, 0.1s -> 0pt100s"""
    s = float(sec[:-1])
    integer_second = int(s)
    if integer_second != 0:
        return f"{integer_second}s"

    # if subsecond
    millis = int(abs(s) * 1000 + 0.5)  # round to nearest ms
    return f"0pt{millis:03d}s"


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


def format_exptime(exptime, type="raw_image"):
    if type == "raw_image":
        return f"{exptime:.1f}s"
    else:
        return f"{exptime:.0f}s"
