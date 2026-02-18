#!/usr/bin/env python
"""
Update PPFLAG in preprocessed science images and master frames
(including those processed with older pipeline versions).

PPFLAG is computed from dependencies in IMCMB:
  - Science: bias | dark | flat (propagated)
  - Master bias: 0 (no master dependencies)
  - Master dark: bias
  - Master flat: bias | flatdark
  - Bit 8: ignored lenient keys when match was found (set by pipeline fetch, not this script)

Usage:
    # Science frames
    python run_update_ppflag.py <science_image.fits> [<science_image2.fits> ...]
    python run_update_ppflag.py <science_dir> [<science_dir2> ...]   # all .fits/.header in dir(s)
    python run_update_ppflag.py --config preproc.yml  # update all science outputs from config

    # Master frames (process bias first, then dark, then flat)
    python run_update_ppflag.py --master master_bias.fits master_dark.fits master_flat.fits
    python run_update_ppflag.py --master <master_frame_dir>         # all frames in that dir
    python run_update_ppflag.py --master --config preproc.yml  # update masters from config

    # Batch by day (directories are expanded to .fits/.header inside)
    for d in /lyman/data2/master_frame/*/; do python run_update_ppflag.py --master "$d"; done
    for d in /lyman/data2/sci_output/*/; do python run_update_ppflag.py "$d"; done

Optional: --dry-run to only print computed PPFLAG without writing.

Real Usage cases:
    find /lyman/data2/master_frame -type d -name "7DT*" -exec python run_update_ppflag.py --master {} \;
"""

import argparse
import os
import sys
from glob import glob

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astropy.io import fits

from pipeline.preprocess import ppflag
from pipeline.utils.header import get_header, write_header_file
from pipeline.path import PathHandler


def _expand_paths(paths: list[str]) -> list[str]:
    """
    Expand path list: replace any directory with the .fits and .header files inside it
    (one path per frame; prefer .fits when both exist). Non-directory paths are kept as-is.
    """
    expanded: list[str] = []

    for p in paths:
        if not p or not isinstance(p, str):
            continue
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            expanded.append(p)
            continue
        # One path per frame: prefer .fits, else .header
        bases: dict[str, str] = {}  # base path (no ext) -> chosen path
        for ext in (".fits", ".header"):
            for f in glob(os.path.join(p, "*" + ext)):
                base = f[: -len(ext)]
                if base not in bases:
                    bases[base] = f
        for base in sorted(bases.keys()):
            expanded.append(bases[base])
    return expanded


def _normalize_imcmb_value(value: str) -> str:
    """Strip whitespace and optional FITS-style quotes from an IMCMB header value."""
    if not value or not isinstance(value, str):
        return ""
    s = str(value).strip()
    while len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s


def _update_header_ppflag(path: str, val: int, dry_run: bool, label: str = "ok") -> bool:
    """Write PPFLAG to header (.header or .fits). Returns True on success."""
    if dry_run:
        return True
    base = path.replace(".header", "").replace(".fits", "")
    header_path = base + ".header"
    fits_path = base + ".fits"
    if os.path.exists(header_path):
        header = fits.Header.fromtextfile(header_path)
        ppflag.set_ppflag_in_header(header, val)
        write_header_file(header_path, header)
    elif os.path.exists(fits_path):
        with fits.open(fits_path, mode="update") as hdul:
            ppflag.set_ppflag_in_header(hdul[0].header, val)
            hdul.flush()
    else:
        return False
    print(f"  [{label}] {path} -> PPFLAG={val}")
    return True


def _is_masterframe_basename(value: str) -> bool:
    """True if the IMCMB value looks like a master frame (bias_*, dark_*, flat_*), not a raw single exposure."""
    b = os.path.basename(_normalize_imcmb_value(value))
    return b.startswith("bias_") or b.startswith("dark_") or b.startswith("flat_")


def _get_all_master_dependencies(path: str) -> list[str]:
    """
    Get all master frame dependency paths from IMCMB.
    Only IMCMB entries that are master frames (bias_*, dark_*, flat_*) are used;
    raw single-exposure paths are ignored.
    Returns list of resolved absolute paths.
    """
    header = get_header(path)
    all_candidates = [v for k, v in sorted(header.items()) if k.startswith("IMCMB")]
    master_candidates = [c for c in all_candidates if _is_masterframe_basename(c)]

    def resolve(c: str) -> str | None:
        """Resolve master frame path using PathHandler."""
        if not c:
            return None
        c = _normalize_imcmb_value(c)
        if not c:
            return None
        # If already an absolute path that exists, use it directly
        if os.path.isabs(c) and os.path.exists(c):
            return os.path.abspath(c)
        # Use PathHandler to resolve the absolute path from basename
        try:
            resolved = PathHandler(c).preprocess.masterframe
            # PathHandler returns str | list[str], normalize to str
            if isinstance(resolved, list):
                resolved = resolved[0] if resolved else None
            if resolved and os.path.exists(resolved):
                return os.path.abspath(resolved)
            elif resolved:
                print(f"  [error] PathHandler resolved '{c}' to '{resolved}' but file does not exist")
            else:
                print(f"  [error] PathHandler could not resolve '{c}' to a valid path")
        except Exception as e:
            print(f"  [error] PathHandler failed to resolve '{c}': {e}")
        return None

    resolved_paths = []
    for c in master_candidates:
        resolved = resolve(c)
        if resolved:
            resolved_paths.append(resolved)
    return resolved_paths


def update_ppflag(path: str, dry_run: bool = False) -> int | None:
    """
    Compute and optionally write PPFLAG for any image (master frame or science).
    Resolves all master frame dependencies from IMCMB and propagates their PPFLAGs
    via bitwise OR. Masters already carry bits 0–4 and 8 from pipeline fetch; this
    script just propagates.

    Returns the computed PPFLAG, or None on error.
    """
    if not os.path.exists(path) and not os.path.exists(path.replace(".fits", ".header")):
        print(f"  [skip] {path}: file not found")
        return None

    # Get all master frame dependencies from IMCMB
    master_paths = _get_all_master_dependencies(path)
    if not master_paths:
        # No master dependencies means this is a master bias (PPFLAG = 0)
        val = 0
    else:
        # Propagate PPFLAG from each master (raise if any ingredient lacks PPFLAG)
        val = ppflag.propagate_ppflag(
            *[ppflag.get_ppflag_from_header(p, raise_if_missing=True) for p in master_paths]
        )

    if dry_run:
        print(f"  [dry-run] {path} -> PPFLAG={val}")
        return val

    if not _update_header_ppflag(path, val, dry_run):
        print(f"  [skip] {path}: no .header or .fits file found")
        return None
    return val


def _collect_paths_from_config(config_path: str, masters: bool) -> list[str]:
    """Collect science or master frame paths from preproc config."""
    from pipeline.config import PreprocConfiguration

    cfg = PreprocConfiguration.from_yaml(config_path)
    paths: list[str] = []
    if masters:
        if hasattr(cfg, "input") and getattr(cfg.input, "masterframe_images", None):
            from pipeline.path import PathHandler
            from pipeline.utils import atleast_1d, flatten

            bdf = flatten(cfg.input.masterframe_images)
            if bdf:
                ph = PathHandler(bdf)
                mf = ph.preprocess.masterframe
                # mf can be [bias, dark, flat] or list of those
                flat_mf = flatten(mf) if mf is not None else []
                for p in flat_mf:
                    if p and isinstance(p, str) and p not in paths:
                        paths.append(p)
        if not paths and hasattr(cfg, "input") and hasattr(cfg.input, "science_images"):
            sci = getattr(cfg.input, "science_images", [])
            if sci:
                from pipeline.path import PathHandler
                from pipeline.utils import atleast_1d, flatten

                ph = PathHandler(atleast_1d(sci))
                mf = ph.preprocess.masterframe
                flat_mf = flatten(mf) if mf is not None else []
                for p in flat_mf if isinstance(flat_mf, list) else [flat_mf]:
                    if p and isinstance(p, str) and p not in paths:
                        paths.append(p)
    else:
        if hasattr(cfg, "input") and hasattr(cfg.input, "science_images"):
            sci = getattr(cfg.input, "science_images", [])
            if sci:
                from pipeline.path import PathHandler
                from pipeline.utils import atleast_1d

                # Use full paths (not basenames) so PathHandler resolves correctly
                sci_full = [os.path.abspath(s) if not os.path.isabs(s) else s for s in atleast_1d(sci)]
                ph = PathHandler(sci_full)
                out = ph.preprocess.processed_images
                paths = atleast_1d(out) if out is not None else []
    return paths


def main():
    ap = argparse.ArgumentParser(description="Update PPFLAG in preprocessed science images and master frames")
    ap.add_argument("images", nargs="*", help="Image path(s)")
    ap.add_argument("--config", "-c", help="Preprocess config YAML; collect paths from config")
    ap.add_argument("--master", "-m", action="store_true", help="Treat inputs as master frames (bias, dark, flat)")
    ap.add_argument("--dry-run", action="store_true", help="Print PPFLAG without writing")
    args = ap.parse_args()

    paths: list[str] = []
    if args.config:
        paths = _collect_paths_from_config(args.config, masters=args.master)
        if not paths:
            print("No paths found from config")
            sys.exit(1)
    else:
        paths = _expand_paths(args.images)
        # Filter out empty directories (warn but continue)
        if not paths and args.images:
            # Check if any input was a directory that had no files
            for img in args.images:
                if os.path.isdir(img):
                    print(f"  [skip] {img}: directory contains no .fits or .header files")
            if not any(os.path.isfile(img) for img in args.images):
                # All inputs were directories with no files
                return

    if not paths:
        if not args.images:
            ap.print_help()
            sys.exit(1)
        # Some inputs were provided but expanded to nothing - already warned above
        return

    # For masters: process in dependency order (bias, dark, flat) so PPFLAG propagates
    if args.master:

        def _master_type(path: str) -> str | None:
            """Return 'bias', 'dark', or 'flat' if path looks like a master frame, else None."""
            b = os.path.basename(path)
            if b.startswith("bias_"):
                return "bias"
            if b.startswith("dark_"):
                return "dark"
            if b.startswith("flat_"):
                return "flat"
            return None

        by_type: dict[str, list[str]] = {"bias": [], "dark": [], "flat": []}
        for p in paths:
            t = _master_type(p)
            if t and t in by_type:
                by_type[t].append(p)
        ordered = by_type["bias"] + by_type["dark"] + by_type["flat"]
        if not ordered:
            ordered = paths
        for p in ordered:
            update_ppflag(p, dry_run=args.dry_run)
    else:
        for p in paths:
            update_ppflag(p, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
