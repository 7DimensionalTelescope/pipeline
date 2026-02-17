#!/usr/bin/env python
"""
Update PPFLAG in preprocessed science images and master frames
(including those processed with older pipeline versions).

PPFLAG is computed from dependencies in IMCMB:
  - Science: bias | dark | flat
  - Master bias: 0 (no master dependencies)
  - Master dark: bias
  - Master flat: bias | flatdark

Usage:
    # Science frames
    python run_update_ppflag.py <science_image.fits> [<science_image2.fits> ...]
    python run_update_ppflag.py --config preproc.yml  # update all science outputs from config

    # Master frames (process bias first, then dark, then flat)
    python run_update_ppflag.py --master master_bias.fits master_dark.fits master_flat.fits
    python run_update_ppflag.py --master --config preproc.yml  # update masters from config

Optional: --dry-run to only print computed PPFLAG without writing.
"""

import argparse
import os
import sys

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astropy.io import fits

from pipeline.preprocess.utils import get_zdf_from_header_IMCMB
from pipeline.preprocess import ppflag
from pipeline.utils.header import get_header, write_header_file
from pipeline.utils.filesystem import swap_ext


def _resolve_master_path(candidate: str, science_path: str) -> str | None:
    """Resolve master frame path from IMCMB value (may be basename or full path)."""
    if not candidate or not isinstance(candidate, str):
        return None
    candidate = str(candidate).strip()
    # Already a valid path
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    # Try relative to science image dir
    sci_dir = os.path.dirname(os.path.abspath(science_path))
    joined = os.path.join(sci_dir, candidate)
    if os.path.exists(joined):
        return os.path.abspath(joined)
    # Try basename in science dir
    bname = os.path.basename(candidate)
    joined = os.path.join(sci_dir, bname)
    if os.path.exists(joined):
        return os.path.abspath(joined)
    # Search recursively from science dir and parents (master frames often in .../nightdate/unit/)
    from glob import glob

    search_dir = sci_dir
    for _ in range(6):
        if not search_dir:
            break
        pattern = os.path.join(search_dir, "**", bname)
        matches = glob(pattern)
        if matches:
            return os.path.abspath(matches[0])
        search_dir = os.path.dirname(search_dir)
    return None


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


def _get_master_dependencies(path: str, ref_dir: str) -> tuple[str | None, str | None]:
    """
    Get dependency paths for a master frame from IMCMB.
    Returns (bias_path, flatdark_path). flatdark_path is None for bias and dark.
    """
    header = get_header(path)
    candidates = [v for k, v in sorted(header.items()) if k.startswith("IMCMB")]
    if not candidates:
        return None, None
    # Resolve with ref_dir as base for search
    def resolve(c: str) -> str | None:
        return _resolve_master_path(c, path) if c else None
    # Master dark: IMCMB001 = bias
    # Master flat: IMCMB001 = bias, IMCMB002 = flatdark
    # Master bias: no master deps
    bias = resolve(candidates[0]) if len(candidates) >= 1 else None
    flatdark = resolve(candidates[1]) if len(candidates) >= 2 else None
    return bias, flatdark


def _master_type(path: str) -> str | None:
    """Return 'bias', 'dark', or 'flat' if path looks like a master frame, else None."""
    b = os.path.basename(path).lower()
    if "master_bias" in b or "mbias" in b:
        return "bias"
    if "master_dark" in b or "mdark" in b:
        return "dark"
    if "master_flat" in b or "mflat" in b:
        return "flat"
    return None


def update_ppflag_for_master_frame(path: str, dry_run: bool = False) -> int | None:
    """
    Compute and optionally write PPFLAG for a master frame.
    - Bias: PPFLAG = 0
    - Dark: PPFLAG = bias PPFLAG (from IMCMB001)
    - Flat: PPFLAG = bias | flatdark (from IMCMB001, IMCMB002)

    Returns the computed PPFLAG, or None on error.
    """
    mtype = _master_type(path)
    if not mtype:
        print(f"  [skip] {path}: not recognized as master bias/dark/flat")
        return None
    if not os.path.exists(path) and not os.path.exists(path.replace(".fits", ".header")):
        print(f"  [skip] {path}: file not found")
        return None

    val: int
    if mtype == "bias":
        val = 0
    else:
        bias_path, flatdark_path = _get_master_dependencies(path, os.path.dirname(path))
        if bias_path is None:
            print(f"  [skip] {path}: could not resolve bias from IMCMB")
            return None
        bias_pp = ppflag.get_ppflag_from_header(bias_path)
        if mtype == "dark":
            val = bias_pp
        else:
            if flatdark_path is None:
                print(f"  [skip] {path}: could not resolve flatdark from IMCMB")
                return None
            val = ppflag.propagate_ppflag(bias_pp, ppflag.get_ppflag_from_header(flatdark_path))

    if dry_run:
        print(f"  [dry-run] {path} -> PPFLAG={val}")
        return val

    if not _update_header_ppflag(path, val, dry_run):
        print(f"  [skip] {path}: no .header or .fits file found")
        return None
    return val


def update_ppflag_for_science_image(science_path: str, dry_run: bool = False) -> int | None:
    """
    Compute and optionally write PPFLAG for a science image.
    Reads IMCMB from header to get bias, dark, flat paths; resolves and propagates PPFLAG.

    Returns the computed PPFLAG, or None on error.
    """
    try:
        zdf = get_zdf_from_header_IMCMB(science_path)
    except Exception as e:
        print(f"  [skip] {science_path}: cannot read IMCMB - {e}")
        return None

    if len(zdf) < 3:
        print(f"  [skip] {science_path}: IMCMB has fewer than 3 entries")
        return None

    bias_path = _resolve_master_path(zdf[0], science_path)
    dark_path = _resolve_master_path(zdf[1], science_path)
    flat_path = _resolve_master_path(zdf[2], science_path)

    if not all((bias_path, dark_path, flat_path)):
        print(f"  [skip] {science_path}: could not resolve master paths (bias={bias_path}, dark={dark_path}, flat={flat_path})")
        return None

    val = ppflag.compute_ppflag_for_science_image(bias_path, dark_path, flat_path)

    if dry_run:
        print(f"  [dry-run] {science_path} -> PPFLAG={val}")
        return val

    if not _update_header_ppflag(science_path, val, dry_run):
        print(f"  [skip] {science_path}: no .header or .fits file found")
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
                for p in (flat_mf if isinstance(flat_mf, list) else [flat_mf]):
                    if p and isinstance(p, str) and p not in paths:
                        paths.append(p)
    else:
        if hasattr(cfg, "input") and hasattr(cfg.input, "science_images"):
            sci = getattr(cfg.input, "science_images", [])
            if sci:
                from pipeline.path import PathHandler
                from pipeline.utils import atleast_1d

                ph = PathHandler(atleast_1d(sci))
                out = ph.preprocess.sci_output
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
        paths = args.images

    if not paths:
        ap.print_help()
        sys.exit(1)

    # For masters: process in dependency order (bias, dark, flat) so PPFLAG propagates
    if args.master:
        by_type: dict[str, list[str]] = {"bias": [], "dark": [], "flat": []}
        for p in paths:
            t = _master_type(p)
            if t and t in by_type:
                by_type[t].append(p)
        ordered = by_type["bias"] + by_type["dark"] + by_type["flat"]
        if not ordered:
            ordered = paths
        for p in ordered:
            update_ppflag_for_master_frame(p, dry_run=args.dry_run)
    else:
        for p in paths:
            update_ppflag_for_science_image(p, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
