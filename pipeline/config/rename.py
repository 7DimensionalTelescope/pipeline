from __future__ import annotations

# This is a script to modify existing config keys in-place.
# Not part of the routine pipeline.


from pathlib import Path
from tqdm import tqdm
from ..path.generator import iter_config


def update_pipeline_yaml_in_place(yml_path: str | Path) -> None:
    """
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
