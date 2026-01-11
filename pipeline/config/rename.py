from __future__ import annotations

# This is a script to modify existing config keys in-place.
# Not part of the routine pipeline.
# functions are preserved for reference


from pathlib import Path
from tqdm import tqdm
from ..path.generator import iter_config


def update_pipeline_yaml_in_place(yml_path: str | Path) -> None:
    """
    2026-01-10
    Modify the YAML file on disk (in-place), applying these renames:

    Global mapping key renames:
      - combine -> coadd
      - combined_photometry -> coadd_photometry
      - imstack -> imcoadd
      - daily_stack -> daily_coadd

    Contextual renames:
      - input.stacked_image -> input.coadd_image
      - imcoadd.stacked_image -> imcoadd.coadd_image
        (also covers imstack.stacked_image before imstack -> imcoadd)

    Uses ruamel.yaml round-trip loader to preserve formatting/anchors.

    Edge case handling:
      - If a rename old->new is requested but `new` already exists at the same mapping level,
        DELETE the existing `new` key and retry the renaming so that the original `old` block
        is renamed in-place (preserving its position).
      - Prints a warning per file if any such collisions occur.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    yml_path = Path(yml_path)

    if not yml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yml_path}")
    if not yml_path.is_file():
        raise ValueError(f"Not a file: {yml_path}")

    # Record edge-case collisions for this file so we can notify once.
    edge_case_hits: list[str] = []

    GLOBAL_RENAMES = {
        "combine": "coadd",
        "combined_photometry": "coadd_photometry",
        "imstack": "imcoadd",
        "daily_stack": "daily_coadd",
    }

    def _merge_maps_fill_missing(dst: CommentedMap, src: CommentedMap) -> None:
        """Fill keys missing in dst from src (dst wins on conflicts)."""
        for k, v in src.items():
            if k not in dst:
                dst[k] = v

    def _rename_key_preserve(cm: CommentedMap, old: str, new: str) -> bool:
        """
        Rename a key in a CommentedMap, preserving order and key comments.

        If `new` already exists at the same level:
          - delete `new`
          - proceed with rename of `old` -> `new`
          - record the edge case for logging
        """
        if old not in cm:
            return False

        # Edge case: target already exists.
        if new in cm:
            # Drop key-associated comments for the target key (if any) to avoid stale comments.
            if getattr(cm, "ca", None) is not None:
                cm.ca.items.pop(new, None)
            cm.pop(new)
            edge_case_hits.append(f"{old}->{new}")

        keys = list(cm.keys())
        idx = keys.index(old)

        # Preserve key-associated comments if present
        old_comment = None
        if getattr(cm, "ca", None) is not None and old in cm.ca.items:
            old_comment = cm.ca.items.pop(old)

        value = cm.pop(old)
        cm.insert(idx, new, value)

        if old_comment is not None:
            cm.ca.items[new] = old_comment

        return True

    def _resolve_imstack_imcoadd_collision(cm: CommentedMap) -> None:
        """
        If both 'imstack' and 'imcoadd' exist in the SAME mapping:
        - Keep the content that is currently under 'imstack' (correct position),
          rename it to 'imcoadd' at the SAME index.
        - Remove the other 'imcoadd' (typically at bottom).
        - Merge missing keys from the removed 'imcoadd' into the kept one (dst wins).
        """
        if "imstack" not in cm or "imcoadd" not in cm:
            return

        keys = list(cm.keys())
        imstack_idx = keys.index("imstack")

        # Preserve key comments
        imstack_comment = None
        imcoadd_comment = None
        if getattr(cm, "ca", None) is not None:
            if "imstack" in cm.ca.items:
                imstack_comment = cm.ca.items.pop("imstack")
            if "imcoadd" in cm.ca.items:
                imcoadd_comment = cm.ca.items.pop("imcoadd")

        imstack_val = cm.pop("imstack")
        imcoadd_val = cm.pop("imcoadd")

        kept_val = imstack_val

        # Merge: fill missing keys in kept_val using the removed imcoadd_val
        if isinstance(kept_val, CommentedMap) and isinstance(imcoadd_val, CommentedMap):
            _merge_maps_fill_missing(kept_val, imcoadd_val)

        cm.insert(imstack_idx, "imcoadd", kept_val)

        # Keep comments from imstack as primary; if it had none, fall back to imcoadd's comments
        if getattr(cm, "ca", None) is not None:
            if imstack_comment is not None:
                cm.ca.items["imcoadd"] = imstack_comment
            elif imcoadd_comment is not None:
                cm.ca.items["imcoadd"] = imcoadd_comment

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=0)

    with yml_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f)

    def _walk(node, path: tuple[str, ...]) -> None:
        if isinstance(node, CommentedMap):
            # 0) Fix order-sensitive collision FIRST (so we keep the right block location)
            _resolve_imstack_imcoadd_collision(node)

            # 1) Global renames
            for old, new in GLOBAL_RENAMES.items():
                if old in node:
                    _rename_key_preserve(node, old, new)

            # 2) Contextual renames
            if path and path[-1] == "input":
                _rename_key_preserve(node, "stacked_image", "coadd_image")

            if path and path[-1] == "imcoadd":
                _rename_key_preserve(node, "stacked_image", "coadd_image")

            # Recurse
            for k, v in node.items():
                _walk(v, path + (str(k),))

        elif isinstance(node, (CommentedSeq, list, tuple)):
            for item in node:
                _walk(item, path)

        # Scalars: nothing to do

    _walk(data, ())

    if edge_case_hits:
        print(
            f"[pipeline-yaml-rename edge-case] {yml_path}: "
            f"target key already existed; deleted target then renamed in-place: {', '.join(edge_case_hits)}"
        )

    with yml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


def downgrade_info_version_if_2(yml_path: str) -> bool:
    """
    2026-01-11
    Change gpPy v2 configs to Py7DT 1.0.0
    info.version: 2.0.0 -> 1.0.0
    info.file to its absolute path


    In-place edit:
      - Find the 'info:' mapping block
      - If within that block a line 'version: 2.0.0' appears, rewrite to 'version: 1.0.0'
    Returns True if a change was made, False otherwise.

    Notes:
      - Plain text approach (no YAML parsing).
      - Updates only the version key under the 'info:' block.
    """
    with open(yml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    changed_version = False
    in_info = False
    info_indent = None

    # Track where we changed version, and whether we saw a file line in info
    version_line_idx = None
    file_line_idx = None
    file_key_prefix = None  # preserves indentation/prefix up to 'file:'

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Enter info block
        if not in_info:
            if stripped == "info:":
                in_info = True
                info_indent = len(line) - len(line.lstrip(" "))
            continue

        # Leave info block when hitting another top-level-ish mapping key
        if stripped != "":
            cur_indent = len(line) - len(line.lstrip(" "))
            if info_indent is not None and cur_indent <= info_indent and stripped.endswith(":"):
                break  # done scanning info block

        # Look for version in info block
        if stripped.startswith("version:"):
            key, rest = line.split("version:", 1)
            rest_stripped = rest.lstrip(" ")
            value_token = rest_stripped.split(None, 1)[0] if rest_stripped.strip() else ""
            if value_token == "2.0.0":
                remainder = rest_stripped[len(value_token) :]  # keep trailing spaces/comments
                new_rest = " " * (len(rest) - len(rest_stripped)) + "1.0.0" + remainder
                lines[i] = key + "version:" + new_rest
                changed_version = True
                version_line_idx = i
            else:
                version_line_idx = i  # still remember where it is (for insertion point)
            continue

        # Look for file in info block
        if stripped.startswith("file:"):
            file_line_idx = i
            file_key_prefix, rest = line.split("file:", 1)
            # keep original trailing newline style
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = f"{file_key_prefix}file: {yml_path}{newline}"

    # If we changed version, ensure info.file is set to yml_path
    if changed_version:
        if file_line_idx is not None:
            # already replaced during scan
            pass
        else:
            # insert file line right after version line if possible, else right after 'info:'
            insert_at = None
            if version_line_idx is not None:
                insert_at = version_line_idx + 1
                # match indentation of other keys under info (assume 2 spaces deeper than info:)
                info_key_indent = " " * ((info_indent or 0) + 2)
                lines.insert(insert_at, f"{info_key_indent}file: {yml_path}\n")
            else:
                # no version line found; fall back: insert after 'info:' line
                for j, l in enumerate(lines):
                    if l.strip() == "info:":
                        info_key_indent = " " * ((len(l) - len(l.lstrip(" "))) + 2)
                        lines.insert(j + 1, f"{info_key_indent}file: {yml_path}\n")
                        break

        with open(yml_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    return False


if __name__ == "__main__":
    for i in range(15, 32):
        iterator = iter_config(f"2025-12-{i}")
        for f in tqdm(iterator):
            print(f)
            update_pipeline_yaml_in_place(f)

    iterator = iter_config("2026*")
    for f in tqdm(iterator):
        print(f)
        update_pipeline_yaml_in_place(f)
