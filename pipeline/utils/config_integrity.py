from __future__ import annotations

import hashlib
import yaml
from pathlib import Path

from ..const.environ import REF_DIR, ROOT_DIR
from ..errors.errors import PipelineError


configs_to_check = [
    "srcExt/prep.sex",
    "srcExt/main.sex",
    "srcExt/prep.param",
    "srcExt/main.param",
    "srcExt/prep.conv",
    "srcExt/main.conv",
    "scamp_7dt_prep.config",
    "scamp_7dt_main.config",
    "7dt.swarp",
    # "7dt.missfits",
    "qa/masterframe.json",
    "qa/science.json",
    "zeropoints.json",
    "depths.json",
    # base yml
    "storage.yml",
    "preproc_base.yml",
    "sciproc_base.yml",
    "preproc_override_ToO.yml",
    "sciproc_override_ToO.yml",
    "sciproc_override_multiEpoch.yml",
]

configs_to_check = [Path(REF_DIR, f) for f in configs_to_check]
hash_file = Path(REF_DIR, "config_hashes.txt")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_known_hashes(hash_file: Path) -> dict[Path, str]:
    """
    Load known hashes from a text file with lines:
        relative/path/to/file  <sha256>
    Lines starting with '#' or empty lines are ignored.
    """
    mapping: dict[Path, str] = {}
    if not hash_file.is_file():
        raise PipelineError(f"Config hash file not found: {hash_file}")

    for line in hash_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise PipelineError(f"Invalid line in {hash_file}: {line!r}")
        rel_path_str, sha = parts
        mapping[Path(rel_path_str)] = sha.lower()

    return mapping


def verify_config_hashes(
    config_paths: list[Path] = configs_to_check,
    hash_file: Path = hash_file,
) -> None:
    """
    Compute sha256 for each config file and compare against known hashes.

    Raises PipelineError if:
      - a config path is missing from the known list
      - a file is missing
      - a hash mismatch is detected
    """
    project_root = Path(ROOT_DIR)

    known_hashes = load_known_hashes(hash_file)

    errors: list[str] = []

    for cfg in config_paths:
        # store paths relative to project_root to match the hash file
        rel = cfg if cfg.is_absolute() else cfg
        if rel.is_absolute():
            rel = rel.relative_to(project_root)

        if rel not in known_hashes:
            errors.append(f"{rel} is not listed in {hash_file}")
            continue

        full_path = project_root / rel
        if not full_path.is_file():
            errors.append(f"{rel} is missing on disk (expected at {full_path})")
            continue

        actual = compute_sha256(full_path)
        expected = known_hashes[rel]

        if actual.lower() != expected:
            errors.append(
                f"{rel} hash mismatch: expected {expected}, got {actual}. "
                "Config changed: bump up the version in pipeline/version.py and run \n\n"
                "from pipeline.utils.config_integrity import update_config_artifacts; update_config_artifacts()\n\n"
                "before running the pipeline."
            )

    if errors:
        msg = "Config integrity check failed:\n  - " + "\n  - ".join(errors)
        raise PipelineError(msg)

    return True


def _infer_annotation(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "Any"


def gen_config_stubs(yml_path: Path, output_path: Path, root_class: str) -> None:
    """Generate TYPE_CHECKING stub classes from a base config YAML."""
    with open(yml_path) as f:
        data = yaml.safe_load(f) or {}

    lines = [
        "# AUTO-GENERATED — do not edit manually.",
        f"# Source: {yml_path.name}",
        "# Run update_config_artifacts() to regenerate.",
        "from __future__ import annotations",
        "from typing import Any, TYPE_CHECKING",
        "",
        "if TYPE_CHECKING:",
        "    from pipeline.config.base import ConfigNode",
        "",
    ]

    section_classes: dict[str, str] = {}

    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        class_name = "".join(p.capitalize() for p in key.split("_")) + "Node"
        section_classes[key] = class_name
        lines.append(f"    class {class_name}(ConfigNode):")
        for subkey, subval in value.items():
            lines.append(f"        {subkey}: {_infer_annotation(subval)}")
        lines.append("")

    lines.append(f"    class {root_class}(ConfigNode):")
    for key, value in data.items():
        ann = section_classes.get(key, _infer_annotation(value))
        lines.append(f"        {key}: {ann}")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def write_config_hashes(
    overwrite: bool = False,
    hash_file: Path = hash_file,
) -> str:
    """
    Compute SHA256 hashes for the given config files and write them to
    os.path.join(REF_DIR, filename).

    Format (one per line):
        relative/path/to/file  <sha256>

    - If the file already exists and overwrite=False -> raise PipelineError.
    - Returns the Path to the written hash file.

    `The trick` to use it to update config_hashes is doing the following:

    Inside a Jupyter notebook,

    # First:
    import pipeline

    # Then, in a next cell:
    from pipeline.utils.config_integrity import update_config_artifacts
    update_config_artifacts()

    The first import failure prevents import blocking of update_config_artifacts,
    which by itself would trigger the config integrity check again otherwise.
    """
    project_root = Path(ROOT_DIR)

    if hash_file.exists() and not overwrite:
        raise PipelineError(
            f"Hash file already exists: {hash_file}. " "Pass overwrite=True if you really want to regenerate it."
        )

    lines: list[str] = []

    for cfg in configs_to_check:
        # Store paths relative to project root (to be stable across machines)
        rel = cfg
        if rel.is_absolute():
            rel = rel.relative_to(project_root)

        full_path = project_root / rel
        if not full_path.is_file():
            raise PipelineError(f"Config file not found: {full_path}")

        digest = compute_sha256(full_path)
        lines.append(f"{rel.as_posix()} {digest}\n")

    hash_file.write_text("".join(lines))
    return str(hash_file)


def gen_all_stubs() -> None:
    """Regenerate all TYPE_CHECKING stub files from their source YAMLs."""
    config_dir = Path(ROOT_DIR) / "pipeline" / "config"
    gen_config_stubs(
        Path(REF_DIR) / "sciproc_base.yml",
        config_dir / "_sciproc_stubs.py",
        "SciProcNode",
    )
    gen_config_stubs(
        Path(REF_DIR) / "preproc_base.yml",
        config_dir / "_preproc_stubs.py",
        "PreprocNode",
    )


def update_config_artifacts(overwrite: bool = True) -> None:
    """Recompute config hashes and regenerate IDE stubs in one shot."""
    write_config_hashes(overwrite=overwrite)
    gen_all_stubs()
