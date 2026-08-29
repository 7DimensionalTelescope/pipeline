import argparse
import glob
import os

from pipeline.config import CrossFilterConfiguration
from pipeline.const import COADD_DIR, PROCESSED_DIR, PROCESSED_DIR_2, TOO_PROCESSED_DIR, TOO_PROCESSED_DIR_2
from pipeline.const.crossfilter import PHOT7DS_SPEC, WHITE_COADD_SPEC, WHITE_PHOTOMETRY_SPEC
from pipeline.path.name import NameHandler
from pipeline.path.path import CrossFilterPathHandler
from pipeline.run import run_crossfilter_reduction
from pipeline.services.database import RawFrameQuery, free_query
from pipeline.utils import atleast_1d
from pipeline.wrapper import DataReduction


def discover_raw_frames(target, nightdate, target_field):
    query = RawFrameQuery().on_date(nightdate)
    query = {
        "target": query.for_target,
        "tile": query.for_tile,
        "object": query.object_name_contains,
    }[target_field](target)
    query.fetch()
    raw_files = query.files()
    if not raw_files:
        raise RuntimeError(f"No raw frames found for {target_field}={target!r} on {nightdate}")
    return raw_files


def validate_filters(paths, expected_filters):
    discovered = sorted(set(atleast_1d(NameHandler(paths).filter)))
    if expected_filters and discovered != sorted(set(expected_filters)):
        raise RuntimeError(f"Expected filters {sorted(set(expected_filters))}, discovered {discovered}")
    return discovered


def discover_science_configs_db(target, nightdate, is_multi_epoch=False):
    if is_multi_epoch:
        rows = free_query(
            """
            SELECT DISTINCT ON (filter) config_file
            FROM process_status
            WHERE config_type = 'science' AND object = %s AND nightdate IS NULL
              AND config_file LIKE %s AND config_file IS NOT NULL
            ORDER BY filter, updated_at DESC, id DESC
            """,
            (target, f"{COADD_DIR}/{target}/%"),
        )
    else:
        rows = free_query(
            """
            SELECT DISTINCT ON (filter) config_file
            FROM process_status
            WHERE config_type = 'science' AND object = %s AND nightdate = %s
              AND config_file IS NOT NULL
            ORDER BY filter, updated_at DESC, id DESC
            """,
            (target, nightdate),
        )
    configs = sorted(row[0] for row in rows)
    if not configs:
        raise RuntimeError(f"No science configs found in the database for {target!r}, {nightdate!r}")
    missing = [path for path in configs if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Database points to missing science configs: {missing}")
    return configs


def discover_science_configs_filesystem(target, nightdate, args):
    if not args.pipeline:
        search_dirs = [os.path.abspath(args.working_dir or os.getcwd())]
    elif args.multi_epoch:
        search_dirs = [os.path.join(COADD_DIR, target)]
    else:
        roots = [TOO_PROCESSED_DIR, TOO_PROCESSED_DIR_2] if args.too else [PROCESSED_DIR, PROCESSED_DIR_2]
        search_dirs = [os.path.join(root, nightdate, target) for root in roots if root]

    candidates = []
    for directory in search_dirs:
        candidates.extend(glob.glob(os.path.join(directory, "**", "*.yml"), recursive=True))

    by_filter = {}
    for path in candidates:
        try:
            properties = NameHandler(path).config_properties
        except Exception:
            continue
        if properties.get("config_type") != "science" or properties.get("object") != target:
            continue
        if not args.multi_epoch and str(properties.get("nightdate")) != str(nightdate):
            continue
        filter_name = properties.get("filter")
        if filter_name in by_filter:
            raise RuntimeError(
                f"Multiple filesystem science configs found for filter {filter_name}: "
                f"{by_filter[filter_name]}, {path}; pass --science-configs explicitly"
            )
        by_filter[filter_name] = path

    configs = sorted(by_filter.values())
    if not configs:
        raise RuntimeError(f"No science configs found under {search_dirs}")
    return configs


def config_from_science_configs(science_configs, args):
    science_configs = sorted(os.path.abspath(path) for path in science_configs)
    expected_coadds = CrossFilterConfiguration._science_config_coadds(science_configs)
    output_yml = CrossFilterPathHandler(
        expected_coadds,
        working_dir=args.working_dir,
        is_pipeline=args.pipeline,
        is_too=args.too,
        is_multi_epoch=args.multi_epoch,
        config_suffix=args.suffix,
    ).crossfilter.output_yml
    if os.path.exists(output_yml) and not args.overwrite_config:
        config = CrossFilterConfiguration(output_yml)
        existing = sorted(os.path.abspath(path) for path in (config.node.input.science_configs or []))
        if existing != science_configs:
            raise FileExistsError(f"{output_yml} has different parents; use --overwrite-config or --suffix")
        return config
    return CrossFilterConfiguration(
        science_configs,
        working_dir=args.working_dir,
        overwrite=args.overwrite_config,
        is_pipeline=args.pipeline,
        is_too=args.too,
        is_multi_epoch=args.multi_epoch,
        config_suffix=args.suffix,
    )


def config_from_coadd_images(coadd_images, args):
    coadd_images = sorted(os.path.abspath(path) for path in coadd_images)
    output_yml = CrossFilterPathHandler(
        coadd_images,
        working_dir=args.working_dir,
        is_pipeline=args.pipeline,
        is_too=args.too,
        is_multi_epoch=args.multi_epoch,
        config_suffix=args.suffix,
    ).crossfilter.output_yml
    if os.path.exists(output_yml) and not args.overwrite_config:
        config = CrossFilterConfiguration(output_yml)
        existing = sorted(os.path.abspath(path) for path in (config.node.input.expected_coadd_images or []))
        if existing != coadd_images:
            raise FileExistsError(f"{output_yml} has different inputs; use --overwrite-config or --suffix")
        return config
    return CrossFilterConfiguration(
        coadd_images,
        working_dir=args.working_dir,
        overwrite=args.overwrite_config,
        is_pipeline=args.pipeline,
        is_too=args.too,
        is_multi_epoch=args.multi_epoch,
        config_suffix=args.suffix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one strict cross-filter white-image configuration")
    parser.add_argument("target")
    parser.add_argument("nightdate", nargs="?")
    parser.add_argument("--filters", nargs="+", help="Assert the complete expected filter set")
    parser.add_argument("--target-field", choices=("target", "tile", "object"), default="target")
    parser.add_argument(
        "--discovery",
        choices=("db", "filesystem", "raw"),
        default="db",
        help="Find science configs in PostgreSQL/on disk, or build all configs from raw intake",
    )
    explicit = parser.add_mutually_exclusive_group()
    explicit.add_argument("--science-configs", nargs="+")
    explicit.add_argument("--coadd-images", nargs="+")
    parser.add_argument("--suffix")
    parser.add_argument("--working-dir")
    parser.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--too", action="store_true")
    parser.add_argument("--multi-epoch", action="store_true")
    parser.add_argument("--overwrite-config", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--white-only", action="store_true")
    stages.add_argument("--phot7ds-only", action="store_true")
    parser.add_argument("--white-photometry", action="store_true", help="Also run the WhiteCatalog source catalog")
    args = parser.parse_args()

    if not args.pipeline and args.working_dir:
        os.chdir(os.path.abspath(args.working_dir))

    discovery_method = None
    if args.science_configs:
        config = config_from_science_configs(args.science_configs, args)
    elif args.coadd_images:
        config = config_from_coadd_images(args.coadd_images, args)
    elif args.discovery in ("db", "filesystem"):
        if not args.multi_epoch and not args.nightdate:
            parser.error("nightdate is required for daily science-config discovery")
        if args.discovery == "db":
            science_configs = discover_science_configs_db(args.target, args.nightdate, args.multi_epoch)
        else:
            science_configs = discover_science_configs_filesystem(args.target, args.nightdate, args)
        config = config_from_science_configs(science_configs, args)
        discovery_method = f"science_configs_{args.discovery}"
    else:
        if args.multi_epoch or args.too:
            parser.error("raw discovery is not supported for multi-epoch or ToO; use db/filesystem/explicit inputs")
        if not args.nightdate:
            parser.error("nightdate is required for RawFrameQuery discovery")
        raw_files = discover_raw_frames(args.target, args.nightdate, args.target_field)
        validate_filters(raw_files, args.filters)
        reduction = DataReduction(
            list_of_images=raw_files,
            use_db=False,
            is_pipeline=args.pipeline,
            enable_crossfilter=True,
            crossfilter_suffix=args.suffix,
        )
        reduction.create_config(overwrite=args.overwrite_config)
        if len(reduction.crossfilter_configs) != 1:
            raise RuntimeError(f"Expected one cross-filter config, found {reduction.crossfilter_configs}")
        config = CrossFilterConfiguration(reduction.crossfilter_configs[0])

    if discovery_method:
        config.record_discovery(config.node.input.source_raw_images, discovery_method)

    if config.node.name.split("_white", 1)[0] != args.target:
        raise RuntimeError(f"Config target {config.node.name!r} does not match {args.target!r}")
    validate_filters(config.node.input.expected_coadd_images, args.filters)
    print(f"Config: {config.config_file}")
    print(f"Parent coadds ({len(config.node.input.expected_coadd_images)}):")
    for image in config.node.input.expected_coadd_images:
        print(f"  {image}")
    print(f"White image: {config.node.input.white_image}")
    print(f"Database updates: {'enabled' if config.node.settings.is_pipeline else 'disabled'}")

    processes = [] if args.phot7ds_only else [WHITE_COADD_SPEC.name]
    if not args.white_only:
        processes.append(PHOT7DS_SPEC.name)
    if args.white_photometry:
        processes.append(WHITE_PHOTOMETRY_SPEC.name)
    run_crossfilter_reduction(config, processes=processes, overwrite=args.overwrite, is_too=args.too)
