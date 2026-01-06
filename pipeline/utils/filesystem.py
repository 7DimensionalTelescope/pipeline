import os
import shutil
from pathlib import Path

from ..const import FACTORY_DIR


def clean_up_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and its contents
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")


def clean_up_factory():
    clean_up_folder(FACTORY_DIR)


def clean_up_sciproduct(root_dir: str | Path, suffixes=(".log", "_cat.fits", "jpg", ".png")) -> None:
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.is_file() and any(path.name.endswith(s) for s in suffixes):
            try:
                path.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed to delete {path}: {e}") from e


def force_symlink(src, dst):
    """
    Remove the existing symlink `dst` if it exists, then create a new symlink.
    """
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except FileNotFoundError:
        pass

    # print(f"current working directory: {os.getcwd()}")
    # print(f"symlinking {src} to {dst}")
    os.symlink(src, dst)
    # os.symlink(os.path.realpath(src), dst)
    # os.system(f"ln -s {src} {dst}")


def swap_ext(file_path: str | list[str], new_ext: str) -> str:
    """
    Swap the file extension of a given file path.

    Args:
        file_path (str): The original file path.
        new_ext (str): The new extension (with or without a leading dot).

    Returns:
        str: The file path with the swapped extension.
    """
    if isinstance(file_path, list):
        return [swap_ext(f, new_ext) for f in file_path]
    stem, ext = os.path.splitext(file_path)

    new_ext = new_ext if new_ext.startswith(".") else f".{new_ext}"
    return stem + new_ext


def add_suffix(filename: str | list[str], suffix: str | list[str]) -> str | list[str]:
    """
    Add a suffix to the filename before the extension.
    Both filename and suffix can be strings or lists of strings.

    Args:
        filename (str | list[str]): The original filename(s).
        suffix (str | list[str]): The suffix to add.

    Returns:
        str | list[str]: The modified filename(s) with the suffix added.
    """
    if isinstance(suffix, list):
        if len(suffix) == 1:
            suffix = suffix[0]
        else:
            assert len(filename) == len(suffix), "Filename and suffix must have the same length"
            return [add_suffix(f, s) for f, s in zip(filename, suffix)]

    if isinstance(suffix, str):
        return _add_suffix(filename, suffix)

    raise ValueError(f"Invalid suffix type: {type(suffix)}")


def _add_suffix(filename: str | list[str], suffix: str) -> str | list[str]:
    if isinstance(filename, list):
        return [add_suffix(f, suffix) for f in filename]
    stem, ext = os.path.splitext(filename)
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{stem}{suffix}{ext}"


def drop_suffix(filename: str | list[str], suffix: str | None = None):
    """
    Remove a suffix from the filename before the extension.
    If no suffix is provided, remove the last underscore section.

    Args:
        filename (str | list[str]): The original filename(s).
        suffix (str | None): The suffix to remove (without extension).
                             If None, drops the last '_suffix'.

    Returns:
        str | list[str]: The modified filename(s).
    """
    if isinstance(filename, list):
        return [drop_suffix(f, suffix) for f in filename]

    stem, ext = os.path.splitext(filename)

    if suffix is None:
        if "_" in stem:
            stem = stem.rsplit("_", 1)[0]
    else:
        suffix = suffix if suffix.startswith("_") else f"_{suffix}"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return f"{stem}{ext}"


def unique_filename(fpath: str):
    if not os.path.exists(fpath):
        return fpath
    else:
        idx = 0
        while True:
            idx += 1
            fpath_new = add_suffix(fpath, f"{idx}")
            if not os.path.exists(fpath_new):
                break
        return fpath_new


def get_basename(file_path):
    return os.path.basename(file_path)


def read_text_file(file, start_row=0):
    """
    Read a text file and return a list of lines.
    """
    with open(file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f][start_row:]
        return lines
