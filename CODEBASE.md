# Py7DT `pipeline` — Codebase Map for AI Agents

Orientation document for automated agents working in this repository. It describes the
**infrastructure that already exists and must be reused, not rebuilt**. Read this before
writing any code that touches paths, filenames, configs, scheduling, or the databases.

---

## 1. What this package is

`pipeline` (a.k.a. Py7DT) is the operational data-reduction pipeline for the
7-Dimensional Telescope, handling both the nightly survey and rapid
target-of-opportunity (ToO) follow-up.

Data live on NFS-mounted storage servers, currently `lyman` and `balmer`. Raw frames
arrive under **`/lyman/data1/obsdata`**; pipeline outputs are dispatched to
**`/lyman/data2/`** and **`/balmer/data1/`**, selected by `nightdate`. Expect this list to
grow — more volumes on `balmer`, and eventually entirely new servers. Nothing in your
code should assume a fixed number of roots or a fixed prefix; the dispatch lives in one
place (§3.2) and that is where a new disk gets added.

Two non-negotiable design principles run through the whole codebase:

1. **Images are string paths, not objects.** FITS files move through the pipeline as
   `str` paths. Per-image metadata lives in FITS headers; per-run metadata lives in YAML
   configs. There is deliberately no custom image class.
2. **External C programs are the compute engines.** SExtractor, SCAMP, SWarp,
   Astrometry.net, HOTPANTS do the science. Python only orchestrates, wraps, and manages.

Consequence: almost every "helper" an agent might be tempted to write (a filename parser,
a path builder, a directory maker, a config reader, a DB client) **already exists**.

### The checkout you are editing is probably not the one in production

This pipeline runs as a live service. The deployed copy is a **separate checkout under a
separate account and conda environment** from whichever working tree you have been asked
to modify — the `cli/` scripts are shebanged to a stable interpreter path that is not
necessarily yours. So: your edits do not take effect in production until someone deploys
them, and conversely a bug you observe in the running system may not exist in your tree.

Stay inside the repository you were pointed at. Do not read, copy from, or write to other
users' home directories, other checkouts, or the production environment unless you were
explicitly asked to. If a task seems to require it, say so and ask rather than reaching
across. The same restraint applies to the live data volumes and the operational databases:
query them read-only, and never queue work, mutate a production config, or delete
products as a side effect of an investigation.

---

## 2. Vocabulary you must know

| Term | Meaning |
| --- | --- |
| `unit` | Telescope number, `7DT01`…`7DT20`. |
| `nightdate` | Date of the observing night in Chile local time (`datetime − 12h`). The primary time key everywhere. |
| `obj` | Target field, usually an RIS tile name `T00000`–`T28519`. |
| `filter` | `m400`…`m875`, wide `m375w` etc., or broadband `u g r i z`. |
| `single` | One individual exposure (raw or processed). |
| `coadd` | Stack of singles for one (nightdate, obj, filter). |
| `diffim` | Coadd minus reference template. |
| `masterframe` | Median-stacked `bias`/`dark`/`flat`, plus `*sig` sigma images and `bpmask`. |
| `preproc` | Instrumental calibration stage. |
| `sciproc` | Everything after preprocessing, named by `ProcessSpec`: `astrometry`, `single_photometry`, `coadd`, `coadd_photometry`, `subtraction`, `difference_photometry`. |
| `factory` | Scratch directory for intermediate products; safe to delete. |
| `is_pipeline` | `True` = operational run writing to system-wide storage roots. `False` = user/ad-hoc run anchored at `working_dir`/cwd. |
| `is_too` | ToO run: separate storage roots, higher priority, ToO DB tracking. |
| `system queue` | SQLite table `scheduler` used for cross-process job bookkeeping. |

### Grouping keys — the load-bearing concept

`pipeline/const/observation.py:1-13` is the single source of truth. **A group of images,
not a single image, is the unit of parallelization — and that group is exactly one
`Configuration`,** either a `PreprocConfiguration` or a `SciProcConfiguration`. Images
sharing a key set share one YAML file, one logger, one queue entry, one `process_status`
row, and one output directory. Grouping is therefore not a detail of the reduction; it is
what defines a job.

```
INSTRUM_GROUP_KEYS            = unit, n_binning, gain, camera
BIAS_GROUP_KEYS               = nightdate + INSTRUM
DARK_GROUP_KEYS               = BIAS + exptime
FLAT_GROUP_KEYS               = BIAS + filter
SURVEY_SCIENCE_GROUP_KEYS     = obj, filter
TRANSIENT_SCIENCE_GROUP_KEYS  = nightdate, obj, filter
```

`*_LENIENT_KEYS` mark keys that may be relaxed when no exact master frame matches; the
compromise is recorded as the `PPFLAG` bitmask in the FITS header.

`TRANSIENT_SCIENCE_GROUP_KEYS` defines both a `SciProcConfiguration` and the daily output
directory. `SURVEY_SCIENCE_GROUP_KEYS` defines the multi-epoch deep-coadd directory.

---

## 3. The backbone — reuse, never reimplement

| Component | Location | Owns |
| --- | --- | --- |
| `NameHandler` | `pipeline/path/name.py:124` | Parsing/assembling 7DT filenames. |
| `PathHandler` | `pipeline/path/path.py:67` | Every path the pipeline reads or writes. |
| `BaseConfig` / `ConfigNode` | `pipeline/config/base.py:14`, `:264` | YAML-backed run state. |
| `PreprocConfiguration` | `pipeline/config/preprocess.py:19` | Preprocess run state. |
| `SciProcConfiguration` | `pipeline/config/sciprocess.py:23` | Science run state. |
| `DataReduction` | `pipeline/wrapper.py:8` | Top-level entry point. |
| `Blueprint` | `pipeline/services/blueprint.py:19` | Grouping → configs → schedule table. |
| `Scheduler` | `pipeline/services/scheduler.py:17` | Dependency + priority bookkeeping. |
| `QueueManager` | `pipeline/services/queue.py:39` | Subprocess execution and shutdown. |
| `BaseSetup` | `pipeline/services/setup.py:14` | Common `__init__` for all processing modules. |
| `Checker` | `pipeline/services/checker.py:28` | `SANITY` flag, QA criteria, input filtering. |
| `Logger` | `pipeline/services/logger.py:49` | Per-config logging with file locking. |
| error registry | `pipeline/errors/definition.py:11`, `registry.py` | Composite exceptions and numeric codes. |
| `DatabaseHandler` | `pipeline/services/database/handler.py:17` | Postgres writes for status + QA. |
| `RawFrameQuery` | `pipeline/services/database/gwportal.py:1958` | Raw-frame discovery from the DB (supersedes the deprecated `RawImageQuery`, `query.py:315`). |
| constants | `pipeline/const/` | Storage roots, grouping keys, process registry. |
| static config | `ref/` | Base YAMLs, astromatic configs, QA criteria, hashes. |

### 3.1 `NameHandler` — filename ↔ grouping keys

Vectorized parser (`str`, `Path`, or list of either). Splits on `_` rather than regex for
speed; scalar input yields scalar attributes, list input yields list attributes.

- `.type` is a 5-tuple from `_detect_image_type` (`path/name.py:309`):
  `(raw|master|calibrated, bias|dark|flat|science, single|coadded|None, difference|None, image|weight|catalog)`.
  Config `.yml` inputs get a config-flavored tuple from `_detect_config_type` (`:285`).
- Attributes: `unit date hms obj filter exptime nightdate n_binning gain camera basename stem parts abspath`.
- Keys not present in the filename are inferred or lazily fetched from the FITS header
  (`gain`, `camera`, `n_binning`). Pre-2024-02-15 commissioning filenames are resolved via
  the database (`path/db.py: unified_names_from_paths`).
- Name builders (`*_basename`) reassemble keys into a filename for each product kind:
  `raw`, `processed`, `conjugate`, `mbias`/`mdark`/`mflat`, `masterframe`.
- Grouping helpers: `find_calib_for_sci` (`:1095`) picks the calibration frames for a
  science frame, `parse_params` (`:1090`) and `get_grouped_files` (`:983`) group by key
  sets, `pick_type` (`:1011`) filters by type tuple.

### 3.2 `PathHandler` — every path in the pipeline

Constructed from input image paths plus run settings; one instance is created per image
group and propagated through the whole run.

```python
path = PathHandler(input_images, working_dir=None, is_pipeline=True,
                   is_too=False, is_multi_epoch=False, config_file=None)
```

Key facts:

- Settings are frozen in `PathHandlerSettings` (`path.py:36`). Use `.replace(...)`
  (`:140`) to derive a variant; do not mutate.
- Disk roots come from `TopDirs` (`path.py:47`) via the classmethod `top_dirs`
  (`path.py:360`), which dispatches on `nightdate` against `DISK_CHANGE_NIGHTDATE`
  (lyman) and `DISK_CHANGE_NIGHTDATE_2` (balmer). Raw input is always
  `/lyman/data1/obsdata`; outputs currently land on `/lyman/data2/` or `/balmer/data1/`
  depending on the night, and more volumes or servers will be appended here.
  **Never hardcode a storage root** — add a new disk in `TopDirs` + `ref/storage.yml`.
- `select_output_dir` (`:406`) then `define_file_dependent_paths` (`:672`) build all
  per-file directories, dispatched by `name.type` and by `is_pipeline`.
- `__getattr__` (`:196`) forwards unknown attributes to the wrapped `NameHandler`
  (accessible as `path.name`) with caching, so `path.obj`, `path.filter`, `path.nightdate`
  all work directly.
- Syntactic sugar suffixes: `_to_string`, `_to_path`, `_collapse` / `_squeeze` / `_compact`.
- Stage namespaces are lazy `cached_property` sub-objects:
  `path.preprocess` (`:1196`), `path.astrometry` (`:1494`), `path.photometry` (`:1589`),
  `path.imcoadd` (`:1649`), `path.imsubtract` (`:1780`), each with `.factory` and
  `.figures` children where relevant.
- Reversible mapping between raw and processed: `path.conjugate` (`:838`),
  `path.raw_images` (`:863`), `path.processed_images` (`:882`).
- Grouping entry points used by `Blueprint`: `take_raw_inventory` (`:977`) and
  `build_preproc_input` (`:991`).
- `get_bpmask` (`:1167`) resolves the bad-pixel mask for an image or header.

Two mixins in `pipeline/path/mixin.py` silently alter attribute access, which explains
most surprising behavior: `AutoCollapseMixin` collapses a uniform list of paths to a
scalar, and `AutoMkdirMixin` **creates the parent directory of any public path attribute
you read**. `_`-prefixed names are exempt from both, which is why internal code uses
`_masterframe`, `_output_dir`, and friends when it only needs a name.

**Do not search the filesystem.** The tree spans several NFS mounts (`lyman`, `balmer`,
more later) that switch by `nightdate`, so a `glob`/`os.walk` over the pipeline roots is
both slow and incomplete. Ask `PathHandler` for a path or the database for an inventory.
Only the **raw frames are canonical** — every derived product (processed single, coadd,
diff, catalog, master frame) may legitimately be absent, stale, or on a different mount.
Treat "the path exists" as an answer you must check, never assume.

**Do not compute dates either.** Directories are keyed by `nightdate`, which is
deliberately *not* the date in the filename or the header: it is the UTC timestamp minus
**15 hours** (TCSpy convention, `path/utils.py:301`), and master-frame filenames carry
`nightdate + 1` on top of that (`path/utils.py:294`). A `strptime` on a filename lands in
the wrong directory for ~97% of science frames and for every master. `NameHandler`
already encodes the offset, the master-frame exception, the pre-2024 branch, and the
TCSpy anomalies where one night carries several dates; it also reconciles the directory
name against the timestamp when they disagree (`name.py:229-235`). Full treatment in
§3.8.

Read-only traversal of existing products, when you really do need to walk:
`pipeline/path/generator.py` provides `iter_single_images`, `iter_coadd_images`,
`iter_config`, `iter_processed`, `iter_masterframe` — level-by-level globs rather than
one giant recursive scan.

### 3.3 `Configuration` — YAML-backed run state

`BaseConfig` (`config/base.py:14`) loads a YAML template into a nested `ConfigNode` tree.
`ConfigNode.__setattr__` (`base.py:269`) writes the backing dict **and re-serializes the
YAML file immediately**, so in-memory and on-disk state never diverge. Assigning a config
attribute is a disk write; do not do it in a tight loop.

Templates live in `ref/` and are selected by `PathHandler`:
`preproc_base.yml`, `sciproc_base.yml`, plus `*_override_ToO.yml` and
`sciproc_override_multiEpoch.yml`.

`SciProcConfiguration` top-level sections: `flag`, `info`, `settings`, `logging`, `input`,
`astrometry`, `photometry`, `imcoadd`, `imsubtract`. The six `flag:` booleans are exactly
the six `ProcessSpec.name` values and are the reprocessing control surface.

Accepted inputs (`config/sciprocess.py:80`): a list of FITS paths (creates a new config),
a path to an existing `.yml` (loads it), or a dict.

**Loading is safe; touching is not.** Opening an existing config does not rewrite it —
`initialize()` is skipped, and the `fill_missing_from_yaml()` that follows mutates only
the in-memory dict (guarded by `_rebuilding`). The moment you assign to any node,
however, the whole YAML is re-serialized to disk. So this reads:

```python
config = SciProcConfiguration(yml_path, write=False, logger=False)   # no disk write
print(config.node.flag.coadd, config.node.imcoadd.bkgsub_type)
```

and this silently edits an operational config file:

```python
config.node.flag.coadd = False        # ← writes yml_path immediately
```

If you only want to inspect, or you are experimenting on a production config, pass
`write=False` so `write_config()` becomes a no-op unless explicitly forced.

Loading is chatty by default: it creates a `Logger`, opens the paired `*.log`, and emits
several INFO lines. **`logger=False` silences it completely** (a null-handler dummy, no
file opened) — use it for bulk scans and read-only tooling, where the log noise and the
file descriptors both add up. `verbose=False` is the softer option: it keeps the logger
but raises the level to `WARNING`.

Useful classmethods:

- `SciProcConfiguration.user_config(...)` (`:224`) — the documented off-pipeline entry.
- `SciProcConfiguration.reset_config(path)` (`:306`) — rebuild from `sciproc_base.yml`
  keeping only `settings.*` and `input.calibrated_images`. Use this to clear stale
  per-stage state.
- `BaseConfig.base_config()` / `from_dict()` / `extract_single_image_config(i)`.

Every config has a paired `*.log` file at the same stem, wired through `node.logging.file`.

### 3.4 Orchestration

```
raw images arrive
      │
      ▼
DataReduction (wrapper.py:8)                 ← the single entry point
      │  is_pipeline / is_too / is_multi_epoch / overwrite_* enter here
      ▼
Blueprint (services/blueprint.py:19)
      ├─ initialize()      :79   PathHandler.take_raw_inventory → PreprocessGroup / ScienceGroup
      ├─ create_config()   :123  ThreadPool writes every YAML to disk
      └─ create_schedule() :166  astropy Table of rows with priority + dependent_idx
      ▼
Scheduler (services/scheduler.py:17)
      │  system queue (SQLite `scheduler` table) or in-memory astropy Table
      ▼
QueueManager (services/queue.py:39)
      │  subprocess.Popen of pipeline/cli/preprocess and pipeline/cli/data_reduction
      ▼
run_preprocess / run_scidata_reduction (pipeline/run.py:22, :54)
      ▼
preprocess ─► astrometry ─► single_photometry ─► coadd ─►
              coadd_photometry ─► subtraction ─► difference_photometry
```

The sciproc step names above are the `ProcessSpec.name` values from
`const/sciproc.py` — the same strings used as `flag:` keys in the config, as
`-processes` CLI arguments, and as error/progress identifiers. Use them, not the class
names (`Astrometry`, `Photometry`, `ImCoadd`, `ImSubtract`), when referring to a stage;
`Photometry` runs three times under three different `ProcessSpec` names.

`PreprocessGroup` / `ScienceGroup` (`services/utils.py:634`, `:733`) hold the image lists
and lazily materialize their config file via `.create_config()`. `SortedGroupDict`
(`:528`) keeps preprocess groups ahead of science groups and orders by group size.

Schedule row columns (`scheduler.py:19-36` and `blueprint.py:206-222`):
`index, config, config_type, input_type, is_ready, priority, readiness, status,
dependent_idx, pid, kwargs, process_start, process_end`.
Statuses: `Ready`, `Pending`, `Processing`, `Completed`, `Failed`, `Paused`, `Stashed`.

Priorities are integers assigned in `Blueprint.create_schedule` (`blueprint.py:166`,
scheme documented in its docstring). Ordering, rather than the exact numbers: ToO
broadband > ToO medium-band > daily survey > user-input/reprocess, and within each tier
preprocess outranks its dependent science jobs. Concurrency caps live as class constants
on `Scheduler` (`scheduler.py:41-42`) and `QueueManager` (`queue.py:22`); preprocess is
capped low because it is I/O-bound, and jobs above the high-priority threshold pause
everything else. Change the scheme in those two places, not at call sites.

`Scheduler._generate_command` (`:948`) maps `config_type` to a CLI script in
`pipeline/cli/`. Return-code policy is in `const/run.py`: `0` success, `1` failure,
`2` empty input after sanity rejection (a normal outcome, not an error).

Daemons and timers live in `systemd/`: `pipeline-queue.service` runs the queue daemon,
`pipeline-trigger.service` runs `cli/run_trigger` (watches `RAWDATA_DIR` for new files;
a `_ToO` directory suffix marks ToO data), `pipeline-clear-schedules.timer` clears
completed rows daily.

`pipeline/services/pipeline_lock.py` gates operational runs behind the `IS_PIPELINE`
environment variable so an ad-hoc session cannot accidentally queue production work.

> Note: `pipeline/services/task.py` (`Task`, `Priority`) and `Astrometry._submit_task`
> (`astrometry/astrometry.py:1343`) are legacy and not on the live path — `QueueManager`
> has no `add_task`. Do not build on them.

### 3.5 Processing modules

All five subclass `BaseSetup` (`services/setup.py:14`) plus `DatabaseHandler`, `Checker`,
and (for sciproc) `RuntimeVersionMixin`. All take a `Configuration` and expose `run()`,
plus a `from_list(images)` classmethod for direct use.

| Module | Class | External engine |
| --- | --- | --- |
| `preprocess/preprocess.py:32` | `Preprocess` | numpy / CuPy cube arithmetic |
| `astrometry/astrometry.py:58` | `Astrometry` | SExtractor, SCAMP, Astrometry.net `solve-field` |
| `photometry/photometry.py:68` | `Photometry` (+ `PhotometrySingle:321`) | SExtractor, GaiaXP synthetic photometry |
| `imcoadd/imcoadd.py:38` | `ImCoadd` | SWarp |
| `subtract/subtract.py:29` | `ImSubtract` | HOTPANTS |

`pipeline/cuda/` holds optional CuPy kernels (weight maps, FFT convolution, masked-pixel
interpolation, image combination, preprocessing). GPU is an option, not a requirement:
host↔device transfer usually eats the speedup.

`pipeline/const/sciproc.py` defines `ProcessSpec` and `SCIPROCESS_REGISTRY` — the single
source of truth for science-stage **names, ordering, error codes, and progress ranges**,
in that order: `astrometry`, `single_photometry`, `coadd`, `coadd_photometry`,
`subtraction`, `difference_photometry`. Each spec also carries `milestones`, the
intra-stage checkpoints that drive the `progress` column in `process_status`. Adding or
reordering a stage starts here. `pipeline/const/preproc.py` does the same for the two
preprocessing phases, `masterframe` and `data_reduction`.

### 3.6 Errors, logging, QA

**Errors.** Composite exceptions of the form `ProcessError.KindError`, built dynamically
from a registry (`errors/registry.py`, wired in `errors/definition.py`). Numeric code =
`100 × process_code + kind_code`; kind `99` is the `UnknownError` sentinel.

```python
from pipeline.errors import AstrometryError
raise AstrometryError.BadWcsSolutionError("no solution")   # code 231
```

There is one process class per `ProcessSpec` plus a few orchestration-level ones
(`SystemError`, `PathHandlerError`, `ConfigurationError`, …); read the bottom of
`errors/definition.py` for the current list rather than memorizing it. Add new kinds with
`register_kind` there. The classes are generated at import time, so IDE rename and
find-references will not see them — use `grep`.

**Logging.** `Logger` (`services/logger.py:49`) wraps stdlib `logging` with
`LockingFileHandler`, optional Slack forwarding, and stdout/stderr redirection. Every
config gets its own logger writing separate INFO and DEBUG files next to the YAML.
`get_high_level_task_logger` (`:18`) handles orchestrator-level messages.

**QA.** `Checker` (`services/checker.py:28`) owns the boolean `SANITY` header key and
`REJ_PROC` (the process that flipped it). Criteria are JSON in `ref/qa/masterframe.json`
and `ref/qa/science.json`. Modules call `apply_sanity_filter_and_report()` (`:59`) in
`__init__` to drop bad frames and rebuild `self.path`. An `INSPCOMM` header card means a
human decided — never recompute over it. Preprocessing additionally writes `PPFLAG`
(bitmask 0–31) recording how much the master-frame match was compromised
(`preprocess/ppflag.py`).

### 3.7 `tools/` — analysis helpers worth knowing before you write your own

`pipeline/tools/` is a small grab-bag of things that are easy to reinvent badly. It is
not imported by the pipeline itself; reach for it in notebooks, QA scripts, and one-off
investigations. Import by full module path (`from pipeline.tools.table import ...`).

- **`visualization.quickvis(image_path=None, *, data=None, binning=4)`** — one-call look
  at a frame. Bins the array down, applies a zscale stretch, and `imshow`s it; hand it a
  `.fits` path, a raw ndarray, or a `.jpg`/`.png` check-plot path (which it displays
  inline and prints so the path is clickable in the IDE). This is the fastest way to eyeball
  a product; do not hand-roll another zscale block.
- **`ds9.create_ds9_region_file(ra=, dec=, ... | x=, y=, ..., radius=, shape=, color=,
  filename=)`** — write a DS9 `.reg` overlay from either sky coordinates (FK5, radius in
  **arcsec**) or pixel coordinates (radius in **pixels**). The standard way to inspect a
  catalog against its image.
- **`table.match_two_catalogs(sci_tbl, ref_tbl, ...)`** — the workhorse sky crossmatch.
  Defaults to SExtractor's `ALPHA_J2000`/`DELTA_J2000` columns, radius in arcsec, and
  supports `join="inner"|"left"|"right"|"outer"`, signed separation components
  (`sep_components=True`), and Gaia proper-motion correction of the reference catalog to
  the observation epoch (`correct_pm=True, obs_time=...`). Always adds a `separation`
  column in arcsec.
- **`table.match_multi_catalogs(cats, ...)`** — N-way joint crossmatch via a
  friends-of-friends graph, one row per matched group and at most one row per catalog.
  Separations are measured against a pivot catalog or the spherical centroid of the group.
  Use this instead of chaining pairwise matches, which biases toward the first catalog.
- **`table.filter_table(table, conditions)` / `build_condition_mask`** — declarative row
  filtering from `(key, op, value)` triples, with word aliases (`"lower"`, `"upper"`,
  `"equal"`) alongside the usual symbols. Accepts a flat list too.
- **`table.add_id_column`**, **`table.spherical_centroid`** — small utilities.
- **`transform.lupton_asinh(img, sky, noise, ...)`** — asinh stretch for display of
  high-dynamic-range frames.
- **`angle.pa_alignment` / `pa_quadrupole_alignment` / `azimuth_deg_from_center`** —
  circular statistics for position angles, used by the astrometry PSF-elongation QA.

### 3.8 Databases — the authoritative index

Because the filesystem is large, multi-mount, and only partially populated, **the
databases are the authoritative index of what exists and how it was made.** Prefer a
query over a directory scan in essentially every situation.

Everything here is best-effort by design: guard with `is_connected`, and let the pipeline
keep processing when the database is down. Modularity and portability outrank a complete
record.

#### The data model in one paragraph

Raw frames are holy. They are the only immutable, complete, canonical population, and
every other row in the system is ultimately traceable back to them. The pipeline *tries*
to maintain a one-to-one correspondence between a raw frame and its processed single
frame, but that correspondence is aspirational, not guaranteed — frames get rejected by
`SANITY`, stages fail, reprocessing campaigns lag. From the singles upward, images fan
into coadds and then differences, and calibration frames fan into master frames; those
many-to-one relations are exactly what the dependency tables record.

#### The four pipeline tables (Postgres)

| Table | Grain | Answers |
| --- | --- | --- |
| `image_qa` | one row per **image** ever produced | "what does this product look like, and is it good?" |
| `image_qa_dependency` | edge between two `image_qa` rows | "what went into this image / what did this image go into?" |
| `process_status` | one row per **configuration** | "how far did this run get, and did it fail?" |
| `process_status_dependency` | edge between two configs | "what else must rerun if I rerun this?" |

**`image_qa`** (`database/image_qa.py:189`) is the all-in-one catalog of every product the
pipeline has ever emitted — master frames, processed singles, coadds, and differences all
live as equivalent rows in one flat schema, discriminated by `image_type`
(`bias|dark|flat|single|coadd|diff`) and `image_group` (`masterframe|science`). It is
populated by `ImageQATable.from_file` (`:155`), which reads the FITS header and maps
header cards onto columns, so **the header is the source and the table is the index**:
identity (`image_name`, `image_path`, `nightdate`, `unit`, `filter`, `object`, `exptime`,
`date_obs`, `ra`/`dec`), verdicts (`sanity`, `inspectd`, `ppflag`, `err_msgs`),
master-frame statistics (`clipmed`, `clipstd`, `sigmean`, `nhotpix`, `shifted`, …), and
science metrics (`seeing`, `seeingsd`, `ellipmn`, `ul5_5`, `zp_auto`, `ezp_auto`,
`skyval`, `skysig`, plus the astrometric separation and radial-FWHM bins). It is written
in real time as products appear, whereas the wider `gwportal` archive ingests the disk
periodically — so `image_qa` is the fresher, narrower view.

**`image_qa_dependency`** (`database/image_qa_dependency.py:66`) is the image-level
provenance graph: `(derived_image_id, source_image_id, dependency_role)`. Its purpose is
**blast-radius analysis** — given that one master flat turns out to be bad, which
singles, which coadds, and which difference images are contaminated? Edges are not
hand-maintained; `sync()` (`:81`) re-reads the `IMCMB*` / `IMG*` cards of a derived file,
classifies each referenced name through `NameHandler`, and fully replaces that image's
rows. Roles are `bias`/`dark`/`flat` for masters and `single`/`coadded` for calibrated
frames. Raw frames are deliberately dropped: they have no `image_qa` row.

**`process_status`** (`database/process_status.py:101`) tracks configurations rather than
images — one row per `PreprocConfiguration` or `SciProcConfiguration`, holding `name`,
`config_type`, `input_type`, the grouping keys, the `config_file`/`log_file`/
`debug_log_file` paths, `progress` (0–100, from the `ProcessSpec` ranges), `status`,
`pipeline_version`, a single `errors` code, and a `warnings` list of codes. Rows are
created late — at `Preprocess` and at `astrometry`, not at config creation — to avoid a
write storm during bulk blueprinting. This is the table behind the web status page and
the one to query when planning a version-targeted reprocessing campaign.

**`process_status_dependency`** (`database/process_status_dependency.py:32`) is the same
idea one level up: `(derived_config_name, source_config_name, dependency_role)`, so you
can ask **which configurations must rerun if I rerun this one**. It mirrors the
`dependent_idx` edges that `scheduler.db` already holds; the SQLite queue stays the
operational source of truth (it must work with Postgres offline) and this table exists
for querying and impact analysis alongside `process_status`. Edges are keyed by config
*name*, not row id, so pending configs are captured before any `process_status` row
exists. `get_sources()` / `get_derived()` (`:117`, `:129`) walk it in either direction.

#### Access layers

- `BaseDatabase` (`database/base.py:17`) — CRUD base for all four: `create_data`,
  `read_data`, `read_data_by_id`, `read_data_by_params`,
  `read_data_by_params_with_date_range`, `update_data`, `delete_data`, `export_to_table`,
  `export_to_csv`.
- `DatabaseHandler` (`database/handler.py:17`) — the mixin every processing module
  inherits; the only thing pipeline code should call. `create_process_data(config_node)`,
  `update_progress(pct)`, `create_image_qa_data(file, process_status_id)`,
  `create_image_qa_dependencies(file, qa_id)`, `mirror_config_dependencies(edges)`,
  `get_process_status(nightdate)`, `get_image_qa(params, image_type=...)`,
  `add_exception_code`, `reset_exceptions`. Routes to `TooDB` when `is_too=True`; no-ops
  when `use_database=False` or disconnected.
- Credentials come from `.env` via `database/const.py`: `DBNAME`, `DBUSER`, `DBHOST`,
  `DBPORT`, `DBPASSWORD`, plus `GWPORTAL_BASE_URL` / `GWPORTAL_API_KEY` for the REST
  fallback.

#### For ad-hoc analysis, write SQL — do not fight the wrappers

The wrappers exist to serve the *pipeline's* fixed access patterns. For one-off
questions — "which units drifted in zeropoint last week", "how many singles never became
coadds", "every config still on version 1.9.x" — writing SQL against the schema below is
faster and clearer than composing `read_data_by_params` calls.

**Where the tables actually are.** There is one Postgres database, `gwu`, holding *two*
schemas: the four pipeline tables live in schema **`pipeline`**, and the whole
Django/gwportal archive (`survey_*`, `catalog_*`, `facility_*`) lives in **`public`**.
The connection's `search_path` is `pipeline, public`, so unqualified names resolve —
but a discovery query filtered on `table_schema = 'public'` will silently miss all four
pipeline tables. Filter by `table_name`, or ask for both schemas:

```python
free_query("SELECT table_schema, table_name FROM information_schema.tables"
           " WHERE table_schema IN ('pipeline','public') ORDER BY 1, 2")
free_query("SELECT column_name, data_type FROM information_schema.columns"
           " WHERE table_name = %s ORDER BY ordinal_position", ["image_qa"])
```

**Two decoys in `public`.** `pipeline_qa` (281k rows) and `pipeline_process` (66k rows)
carry familiar-looking columns (`clipmed`, `clipstd`, `seeing`, `status`, `progress`) but
they are the *predecessor* generation and are effectively frozen — `pipeline_process`
stops at 2026-01-02 while `pipeline.image_qa` is written continuously. Note the naming
inversion that makes this easy to get wrong: unqualified `image_qa` resolves to
`pipeline.image_qa` (current), while unqualified `pipeline_qa` resolves to
`public.pipeline_qa` (dead). Schema-qualify when in doubt.

**Joining pipeline QA to the archive.** They share a database, so this works in one
query, but **the filenames are not a join key** — `image_qa.image_name` is
`T07940_m575_7DT06_20251128_061019_100s` while
`survey_processedscienceframe.filename` is
`calib_7DT03_T21900_20241020_005010_m500_100.fits`, a different generation's convention
under a different root. Do not try to string-match them.

The real key is **`image_qa.imageid` = `survey_scienceframe.image_id`**, a 32-char hash.
It is populated for ~90% of singles (704k of 785k) and `survey_scienceframe.image_id` is
uniquely indexed, so the join is cheap. This is the shortest path to any question mixing
QA metrics with observing conditions — airmass, weather, guiding RMS, moon separation,
tile and target — none of which `image_qa` stores:

```python
free_query(
    "SELECT q.unit, avg(q.seeing), avg(s.airmass), count(*)"
    " FROM image_qa q JOIN survey_scienceframe s ON s.image_id = q.imageid"
    " WHERE q.image_type = 'single' AND q.sanity AND q.nightdate > %s"
    " GROUP BY q.unit ORDER BY q.unit",
    ["2026-06-01"],
)
```

Rows with `imageid IS NULL` predate the field or are products with no single raw parent;
an inner join silently drops them, so count both sides when completeness matters.

```python
from pipeline.services.database import free_query

rows = free_query(
    "SELECT unit, filter, AVG(zp_auto), COUNT(*) FROM image_qa"
    " WHERE image_type = 'single' AND sanity AND nightdate BETWEEN %s AND %s"
    " GROUP BY unit, filter ORDER BY unit",
    ["2026-05-01", "2026-05-07"],
)
```

`free_query(query, params)` (`database/query.py:107`) runs any statement through the
shared connection pool and returns raw tuples. It is marked *dev only* in the source
because it interpolates nothing for you — **always pass values as `params`, never format
them into the string**, and keep it to `SELECT`s. Writes belong in `DatabaseHandler` so
that timestamps, dependency mirroring, and ToO routing stay consistent.

Columns, so you can write the query without reading the source:

| Table | Columns |
| --- | --- |
| `image_qa` | `id, process_status_id, created_at, updated_at, image_name, image_type, image_group, image_path, imageid, nightdate, unit, filter, object, exptime, date_obs, altitude, azimuth, ra, dec, sanity, inspectd, err_msg, err_msgs, ppflag, trimmed, clipmed, clipstd, clipmin, clipmax, sigmean, shifted, shftscr, edgevar, uniform, nhotpix, ntotpix, seeingmn, seeingsd, pa_quad, pa_align, isep_q2, isep_p95, i_recall, bin0fwhm, bin1fwhm, bin2fwhm, bin0mad, bin1mad, bin2mad, unmatch, rsep_rms, rsep_q2, rsep_p95, awincrmn, ellipmn, zp_5, ezp_5, seeing, peeing, rotang, skyval, skysig, zp_auto, ezp_auto, ul5_5, stdnumb, inf_filt, bkg_step` |
| `process_status` | `id, created_at, updated_at, name, config_type, input_type, pipeline_version, nightdate, unit, filter, object, progress, status, warnings, errors, config_file, log_file, debug_log_file, comments_file` |
| `image_qa_dependency` | `derived_image_id, source_image_id, dependency_role` (ids are `image_qa.id`) |
| `process_status_dependency` | `derived_config_name, source_config_name, dependency_role, created_at` (keyed by config *name*, not id) |

Types that matter when you write predicates:

- `nightdate` is `date` and `date_obs` is a naive-UTC `timestamp`. **They disagree for
  about 97% of science frames** — see the next subsection, which you must read before
  writing any date predicate.
- `sanity` and `inspectd` are real booleans, so `WHERE sanity` suffices.
- `ppflag` is a `smallint` bitmask (observed 0, 1, 4, 5, 8, 9 …) recording which grouping
  key was relaxed. Test bits with `&`; never compare it for equality.
- `process_status.errors` is a single `integer` code; `warnings` is `jsonb` holding a
  list of integer codes, e.g. `[304, 504, 400, 670]`. Empty is `[]`, **not** `NULL`, so
  `WHERE warnings IS NOT NULL` does not filter out clean runs — use
  `jsonb_array_length(warnings) > 0`. Decode both against the registry in
  `errors/definition.py`; the database stores numbers, never names.

Observed `dependency_role` values are `bias`, `dark`, `flat`, and `single` in
`image_qa_dependency`, and only `preprocess` in `process_status_dependency`. The code
also defines `coadded` (`image_qa_dependency.py:58`) but no rows currently carry it —
treat its absence as "not yet backfilled", not as proof that coadds have no parents.

`ProcessStatusTable` (`process_status.py:13`) matches its table exactly, name and order.
**`ImageQATable` (`image_qa.py:16`) does not** — the live table has drifted: it carries
both `err_msg` and `err_msgs`, it spells the dataclass's `clipped` as **`trimmed`**, and
its physical column order differs from the field order because columns were appended by
`ALTER TABLE`. Consequences: name your columns explicitly (never `SELECT *` into
`ImageQATable.from_row` without passing `columns`), and treat `information_schema` — not
the dataclass — as ground truth for `image_qa`. Report the drift rather than
"fixing" either side; the mapping in `from_file` is what production writes through.

Discriminate `image_qa` rows with `image_type` (`bias|dark|flat|single|coadd|diff`) and
`image_group` (`masterframe|science`); most numeric columns are populated for only one of
those families, so filter before you aggregate.

**`process_status.status` is a composite string, not a flat enum.** Most values are
`<process_name>-<state>`, e.g. `astrometry-configured`, `imcoadd-configured`,
`difference_photometry-completed`, alongside bare `pending`, `astrometry`, and
`completed`. A naive `WHERE status = 'completed'` therefore misses the overwhelming
majority of finished configs. Match on the state suffix, or use `progress = 100`.

Doing that hits a psycopg quirk: **a literal `%` must be doubled to `%%` whenever you
pass a `params` list, and left single when you do not.** Mixing them up raises
`ProgrammingError: only '%s', '%b', '%t' are allowed as placeholders`.

#### `nightdate` is a defined quantity, not a date you can compute

**`nightdate` is deliberately not the date in the timestamp, and never derive one from
the other.** The rule is `subtract_half_day` (`path/utils.py:301`):

```
nightdate = (UTC observation timestamp) − 15 hours
```

Fifteen, not twelve — it is the TCSpy convention, chosen so a whole Chilean observing
night lands on one label regardless of where UT midnight falls inside it. Consequences
you will hit immediately:

- `image_qa.date_obs` disagrees with `nightdate` for **762k of 785k singles (97%)**. Night
  `2025-11-27` holds `2025-11-28` timestamps. This is the normal case, not an edge case.
  Group by `nightdate` whenever you mean "a night"; `date_obs::date` is a different and
  almost always wrong quantity.
- **The inverse is not well-defined.** `nightdate + 1` is usually the calendar date but
  can equal the `nightdate` itself, which is exactly why `add_half_day` sits commented
  out in `name.py:232` with the note that it "can mutate the true date crossing midnight".
- **Master frames are worse: their filenames carry `nightdate + 1`**, via `add_a_day`
  (`path/utils.py:294`). A master whose filename says `20251128` belongs to night
  `2025-11-27`. Reading the date out of a master frame's name gives the wrong night.
- Real data violates the model. `name.py:230` notes that some nightdates carry multiple
  dates because of TCSpy errors, and frames before `2024-02-15` take a separate branch
  (`name.py:158`).

**This is a first-order reason to go through `NameHandler` / `PathHandler` rather than
parsing dates yourself.** They already encode the −15 h rule, the master-frame +1 day
offset, the pre-2024 branch, and the TCSpy anomalies, and they reconcile the directory
name against the timestamp when the two disagree (`name.py:229-235`). Any `strptime` on
a filename or `date_obs::date` in a query is reimplementing that, and will be wrong for
a large fraction of the archive rather than for a rare edge case.

#### Column names differ across schemas — map before you join

The pipeline tables use flat, human-readable names. The Django archive uses Django
conventions: integer foreign keys and `_name` suffixes. Same concept, different column,
and sometimes a different type:

| Concept | `pipeline.image_qa` | `public.survey_scienceframe` |
| --- | --- | --- |
| night | `nightdate` (`date`) | `night_id` → `survey_night.date` |
| timestamp | `date_obs` (naive UTC) | `obstime` (`timestamptz`), `local_obstime` |
| unit | `unit` (`'7DT01'`) | `unit_id` → `facility_unit.name` |
| filter | `filter` (`'m650'`) | `filter_id` → `facility_filter.name` |
| target | `object` | `object_name`, plus `target_id` / `tile_id` |
| identity | `imageid` | `image_id` (the join key) |
| file | `image_name`, `image_path` | `original_filename`, `unified_filename`, `file_path` |

Two verified facts that make this workable. `survey_night.date` **agrees with
`image_qa.nightdate` exactly** — all 638,955 joined rows match, no offset — so it is a
safe bridge for night-level questions. But `date_obs` and `obstime` are the same instant
in *different representations*: the session timezone is `Asia/Seoul`, so a frame stored
as naive UTC `2025-11-28 06:10:19` in `image_qa` renders as `2025-11-28 15:10:19+09:00`
in the archive. **Comparing them directly does not error — it returns zero rows**, which
reads like "these tables share no data" rather than "this predicate is wrong." Measured
over the 638,955 joined singles, `q.date_obs = s.obstime::timestamp` matches 0 while
`q.date_obs = s.obstime AT TIME ZONE 'UTC'` matches all of them. Use the latter, or join
on `imageid` and avoid the question entirely.

#### Scale, and why your query is slow

Roughly: `image_qa` 970k rows, `image_qa_dependency` 2.5M, `process_status` 240k,
`process_status_dependency` 306k, covering 2023-10-09 to the present.

**The columns you will naturally filter on are not indexed.** `image_qa` carries only
`(id)`, a partial `(id) WHERE ppflag IS NULL`, and `(process_status_id, image_name)`;
`process_status` carries `(id)` and `(name)`. There is no index on `nightdate`,
`image_type`, `unit`, `filter`, or `object`. So the obvious nightly-QA query above plans
as a **parallel sequential scan over all 970k rows** — a few seconds, not milliseconds.
That is expected behavior, not a broken connection or a wrong query; check with
`EXPLAIN` before concluding anything is wrong, and do not "fix" it by adding indexes to a
production database.

The dependency tables *are* indexed for traversal — `source_image_id` and the
`(derived, source, role)` triple on `image_qa_dependency`, both directions on
`process_status_dependency` — so blast-radius walks are fast even though metric scans are
not. Where you have one, filtering by `process_status_id` or `name` also hits an index.
Prefer one aggregate query over a Python loop issuing many small ones; the scan cost is
paid per query, so a loop over nights is the single easiest way to turn seconds into
minutes.

```python
free_query("SELECT count(*) FROM process_status WHERE status LIKE '%-completed'")

free_query("SELECT count(*) FROM process_status"
           " WHERE status LIKE '%%-completed' AND nightdate > %s", ["2025-11-01"])
```

#### The wider archive: `gwportal`

`database/gwportal.py` wraps the full Django/Postgres archive of every 7DT frame — raw,
processed, combined, their ToO variants, tiles, targets, and master calibration frames —
including products made by the predecessor pipeline. Two interchangeable backends: direct
`psycopg` SQL (default, roughly 20× faster) with automatic fallback to the REST client;
pass `backend="sql"` or `"http"` to force one.

```python
from pipeline.services.database import (
    GWPortalQuery, ProcessedFrameQuery, MasterFlatQuery, TileQuery,
)

rows = GWPortalQuery("processed").query(date_start="2026-05-01", filter_name="m650")
tbl  = GWPortalQuery("processed").query_table(date_start="2026-05-01", filter_name="r")

rows = (ProcessedFrameQuery().on_date("2026-05-01")
        .by_units(["7DT01", "7DT02"]).with_filter("m525").fetch())
flats = MasterFlatQuery().by_unit("7DT01").on_nightdate("2026-04-30").with_filter("g").fetch()
near  = TileQuery().cone_search(ra=180.0, dec=0.0, radius=2.0).fetch()
```

#### Seeding a run: prefer `RawFrameQuery` over `RawImageQuery`

**Write new code against `RawFrameQuery`** (`gwportal.py:1958`). `RawImageQuery`
(`query.py:315`) is explicitly marked deprecated in the source and is being retired; it
still exists because `run.py:query_observations` and the operational entry points have
not been migrated yet. Do not delete it, do not extend it, and do not reach for it in
anything new.

`RawFrameQuery` covers the same ground with the shared fluent grammar used by every other
`gwportal` builder — `.on_date()`, `.between()`, `.by_unit()/.by_units()`,
`.with_filter()`, `.for_target()`, `.cone_search()`, plus `.where(**anything)` as an
escape hatch — and terminates with `.fetch()` (list of dicts), `.fetch_table()` (astropy
`Table`), or `.files()` (bare paths). It also has `.nightdates()`, which answers "which
nights exist for this unit/filter?" without pulling any frames, and `.sql()` for
inspecting the generated query.

```python
from pipeline.services.database import RawFrameQuery

files = (RawFrameQuery().on_date("2026-05-05").by_unit("7DT01")
         .with_filter("m650").files())

nights = RawFrameQuery().by_unit("7DT01").between("2026-05-01", "2026-05-31").nightdates()
tbl    = RawFrameQuery().for_target("T02386").fetch_table()
```

The one thing `RawFrameQuery` does not replicate is `RawImageQuery`'s auto-classified
parameter list (`["2026-05-05", "7DT01", "m650"]` sniffed into date/unit/filter by regex),
which is what `DataReduction(input_params=[...])` passes through
`query_observations` (`pipeline/run.py:122`). That path also falls back to
`query_observations_manually` — filesystem globbing — when the database is unreachable;
it is the slow degraded path, not the intended one.

#### The two SQLite stores

- **`/var/db/scheduler.db`** (`SCHEDULER_DB_PATH`), table `scheduler` — the system queue.
  Cross-process job state, dependencies, PIDs. Reach it through `Scheduler`, not raw SQL.
- **`/var/db/too_requested.db`** (`TOO_DB_PATH`) — one row per ToO request: trigger /
  observation / transfer / processed timestamps, progress, output file list, and which of
  the three notification emails have been sent. Also what makes ToO backfill possible
  without manual bookkeeping (`services/database/too.py`, `too/backfill.py`).

### 3.9 Constants, static config, and versioning

- `pipeline/const/environ.py` reads `ref/storage.yml` and then `.env` (which overrides).
  All storage roots, reference-catalog directories, log paths, and DB/socket paths are
  defined here. `pipeline/const/observation.py` holds grouping keys, filter tables, and
  `HEADER_KEY_MAP`.
- `ref/` also holds the astromatic configs (`srcExt/*.sex|param|conv`,
  `scamp_7dt_*.config`, `7dt.swarp`), QA criteria, and `config_hashes.txt`.

#### Version policy — bump for science, not for size

`pipeline/version.py` follows a rolling release with the usual three digits, but the last
digit carries an extra meaning: **bump the patch digit for any change that alters how
data is processed, no matter how small the diff.** A one-line threshold change in a
`.sex` file, a different SWarp kernel, a new rejection criterion — all of these are
version bumps, because the version is the only handle anyone has for asking "which code
produced this file?" later. Pure refactors, comments, and logging changes are not.

#### Versioning is per-process, and it can silently trigger overwrites

The version is stamped everywhere: `info.creation_version` and `info.runtime_version` on
the config, **a separate `runtime_version` inside each process's own config section**
(`RuntimeVersionMixin.record_runtime_version`, `services/version_check.py:32`), and the
pipeline version on every `process_status` row. The per-section stamp is the important
one — `astrometry`, `photometry`, `imcoadd`, and `imsubtract` each carry their own
recorded version, so "this file is up to date" is a per-stage question, never a
whole-config one.

The minimums are per-process too: `MIN_SCIPROC_RUNTIME_VERSION_MAP` (`version.py:9`) maps
config section → oldest acceptable version, with `MIN_SCIPROC_RUNTIME_VERSION` and
`MIN_PREPROC_RUNTIME_VERSION` as the overall floors. On every stage,
`RuntimeVersionMixin.resolve_overwrite` (`version_check.py:10`) compares the recorded
version against that stage's minimum and **escalates `overwrite=True` on its own** when
it is older. Raising one entry in that map is therefore how you force a fleet-wide
recompute of exactly one stage, without touching a single call site.

> **This is a data-destroying mechanism. Treat it as one.**
>
> - `is_below_min` (`version.py:24`) returns `True` when the recorded version is
>   **missing or unparseable**, not only when it is old. An unstamped or hand-edited
>   config is indistinguishable from an ancient one, and gets overwritten.
> - The escalation is per-stage, so a reprocess can cut through the *middle* of a chain:
>   re-running `astrometry` under new WCS while the existing coadd and subtraction
>   products stay behind leaves a config whose products no longer descend from each
>   other. Downstream stages must be re-run, in order, or explicitly invalidated.
> - Reprocessing is not guaranteed idempotent. Running the same stage twice can yield
>   different results — masters may be re-matched leniently, headers accumulate, and
>   inputs already replaced on disk are not the inputs the first run saw.
> - Before any reprocessing campaign, work out the blast radius from
>   `image_qa_dependency` / `process_status_dependency` (§3.8) rather than guessing.
>
> **If the recorded-version bookkeeping looks incomplete or inconsistent for the data you
> are about to touch — missing `runtime_version` stamps, a section version newer than
> `info.runtime_version`, `process_status` disagreeing with the YAML — stop, report what
> you found, and do not run anything that writes or overwrites data.** Ask first. Getting
> this wrong quietly corrupts months of archive.

#### The hash guard, and how to refresh it

`pipeline/utils/config_integrity.py` SHA-256-checks every scientifically meaningful file
in `ref/` **at import time** (`pipeline/__init__.py`). Change any of them and *every*
`import pipeline` raises `PipelineError` until you bump the version and regenerate the
hashes. This is deliberate: it makes it impossible to quietly alter processing behavior
without leaving a version trail.

The refresh has a bootstrap problem — you cannot import the helper, because importing it
triggers the very check that is failing. The documented workaround is to let the first
import fail, which marks the guard as already-run, then import the helper in a **separate
cell or statement**:

```python
# Step 1 — in its own cell / statement. This is EXPECTED to raise PipelineError.
import pipeline

# Step 2 — now the guard has already fired, so this import succeeds.
from pipeline.utils.config_integrity import update_config_artifacts
update_config_artifacts()          # rewrites ref/config_hashes.txt + regenerates stubs
```

`update_config_artifacts()` (`config_integrity.py:244`) does two things: `write_config_hashes(overwrite=True)`
rewrites `ref/config_hashes.txt`, and `gen_all_stubs()` regenerates
`config/_sciproc_stubs.py` and `config/_preproc_stubs.py` from the base YAMLs so IDE
autocompletion tracks the new schema. Run it after **any** edit to a file in
`configs_to_check` (`config_integrity.py:11`) — the `srcExt/*` SExtractor configs, the
SCAMP and SWarp configs, `qa/*.json`, `zeropoints.json`, `depths.json`, `storage.yml`, and
all the base/override YAMLs.

Order of operations, every time: **edit the `ref/` file → bump `__version__` → run
`update_config_artifacts()` → commit all three together.**

---

## 4. Where things actually live

Defaults from `ref/storage.yml`; `.env` overrides them, and `pipeline.const` is the only
place code should read them from.

| What | Path |
| --- | --- |
| Raw frames (canonical) | `/lyman/data1/obsdata/<unit>/<nightdate>_gain<g>[_ToO]/` |
| Processed products | `/lyman/data2/processed` → `/balmer/data1/processed` |
| Master frames | `/lyman/data2/master_frame` → `/balmer/data1/master_frame` |
| Factory (scratch) | `/lyman/data2/factory` → `/balmer/data1/factory` |
| Multi-epoch deep coadds | `/lyman/data2/coadd` |
| ToO products / factory | `/lyman/data2/too`, `/lyman/data2/too_factory` |
| Disk switch by `nightdate` | `2026-04-08` (lyman→balmer), `2027-01-10` (next) |
| System queue | `/var/db/scheduler.db` (table `scheduler`) |
| ToO database | `/var/db/too_requested.db` |
| Queue wake socket | `/run/queue/queue.sock` |
| Pipeline logs | `/var/log/pipeline/`, `high_level_tasks.log` |
| Trigger log | `/var/log/pipeline-trigger.log` |
| Astrometric refcats | `/lyman/data2/py7dt_requisites/ref_scamp/` (RIS tiles, custom, queried) |
| Photometric refcats | `/lyman/data1/factory/ref_cat` (GaiaXP synphot per tile) |
| Reference (template) images | `/lyman/data1/factory/ref_frame` |
| Static configs, QA criteria, hashes | `ref/` in this repo |

Per-file layout, pipeline mode (`is_pipeline=True`):

```
<PROCESSED_DIR>/<nightdate>/<obj>/<filter>/
    singles/    *.fits, *_cat.fits, *_weight.fits
    coadd/      *_coadd.fits, *_coadd_cat.fits
    difference/ *_diff.fits, *_diff_cat.fits
    figures/    astrometry/ imcoadd/ imsubtract/ photometry/
    <obj>_<filter>_<nightdate>.yml   and matching .log
<FACTORY_DIR>/<nightdate>/<obj>/<filter>/   scratch — safe to delete
<MASTER_FRAME_DIR>/<nightdate>/<unit>/      bias/dark/flat/*sig/bpmask
<COADD_DIR>/<obj>/<filter>/                 multi-epoch deep coadds
```

User mode (`is_pipeline=False`): everything is anchored at `working_dir` or cwd, with
`tmp/` as the factory. See the ASCII trees in `pipeline/path/const.py`.

Directory-name constants are in `pipeline/path/const.py` — use them, don't inline strings.

---

## 5. Where to look for a given task

| Task | Go to |
| --- | --- |
| Parse or build any 7DT filename | `pipeline/path/name.py` |
| Locate any input, output, or intermediate file | `pipeline/path/path.py` (+ stage sub-classes) |
| Add a new output product path | the relevant `Path*` class in `path/path.py`, then use it everywhere |
| Change a storage root or add a disk | `ref/storage.yml`, then `PathHandler.top_dirs` (`path.py:360`) |
| Change how images are grouped | `pipeline/const/observation.py`, then `NameHandler.find_calib_for_sci` |
| Add or change a config parameter | `ref/sciproc_base.yml` or `ref/preproc_base.yml`, bump version, `update_config_artifacts()` (regenerates hashes *and* stubs) |
| Inspect a config without touching it | `SciProcConfiguration(yml, write=False, logger=False)` |
| Refresh `ref/config_hashes.txt` after editing `ref/` | `update_config_artifacts()` — see the two-step bootstrap in §3.9 |
| Add a science stage | `const/sciproc.py` `ProcessSpec` → `run.py:run_scidata_reduction` → `flag:` in `sciproc_base.yml` |
| Change job priority or concurrency | `services/blueprint.py:166` (priorities), `services/scheduler.py:41-42` (caps) |
| Debug a stuck or failed job | `scheduler.db` via `Scheduler`, `cli/rerun_failed_tasks`, `cli/terminate_scheduler_tasks` |
| Change master-frame selection | `preprocess/preprocess.py:_fetch_masterframe` (`:567`), `preprocess/ppflag.py` |
| Change bad-pixel-mask criteria | `preproc_base.yml: preprocess.n_sigma`, `preprocess.py:update_bpmask` (`:924`) |
| Change the WCS solution or its QA | `astrometry/astrometry.py` (`run_scamp:895`, `run_solve_field:784`, `evaluate_solution:1123`) |
| Change zeropoint calibration | `photometry/photometry.py:PhotometrySingle.calculate_zp` (`:892`) |
| Change aperture or detection settings | `sciproc_base.yml: photometry.sex_vars`, `ref/srcExt/*` |
| Change background subtraction or weights | `imcoadd/imcoadd.py:bkgsub` (`:332`), `calculate_weight_map` (`:524`) |
| Change coaddition/resampling | `imcoadd/imcoadd.py:reproject_and_coadd_with_swarp` (`:915`), `ref/7dt.swarp` |
| Change reference-image selection | `subtract/subtract.py:find_reference_image` (`:169`) |
| Change a sanity/QA threshold | `ref/qa/science.json`, `ref/qa/masterframe.json` (hash-checked) |
| Add an error type | `pipeline/errors/errors.py` + `register_kind` in `errors/definition.py` |
| Write/read pipeline status or QA rows | `services/database/handler.py` |
| Find raw images for a night | `RawFrameQuery` (new code) / `run.py:query_observations` (existing path) |
| Query the full archive (any frame type, any era) | `services/database/gwportal.py` |
| "What was contaminated by this bad master frame?" | `image_qa_dependency` (`get_derived` recursively) |
| "What must I rerun if I rerun this config?" | `process_status_dependency`, or `scheduler.db` `dependent_idx` |
| Add a QA metric to the DB | new column + field on `ImageQATable` (`image_qa.py:16`); `from_file` maps it from the header automatically |
| Walk existing products (last resort) | `pipeline/path/generator.py` |
| Look at an image, or overlay a catalog on it | `tools/visualization.py:quickvis`, `tools/ds9.py:create_ds9_region_file` |
| Crossmatch catalogs, filter a table | `tools/table.py` (`match_two_catalogs`, `match_multi_catalogs`, `filter_table`) |
| ToO behavior | `too/`, `config/toodb.py`, `services/database/too.py` |
| Runnable examples | `run_examples/` |

---

## 6. Invariants — violating these breaks things

1. **Never write a new filename parser or path builder.** Use `NameHandler` /
   `PathHandler`. They are the adaptation point for a different facility.
2. **Never hardcode a storage path and never scan the filesystem to find products.**
   Import roots from `pipeline.const`, ask `PathHandler` for a path, and ask the database
   for an inventory. Roots switch by `nightdate` across NFS mounts.
3. **Only raw frames are guaranteed to exist.** Every derived product may be missing or
   stale; the raw↔processed correspondence is a goal, not an invariant. Check, don't
   assume.
4. **Accessing a public `PathHandler` attribute creates directories.** Use the
   `_`-prefixed variant when you only need the name (grouping, existence checks, lookup
   templates).
5. **Assigning to a `ConfigNode` attribute writes YAML to disk.** Loading is free;
   touching is not. Pass `write=False` to inspect, `logger=False` to load quietly.
6. **Any change to how data is processed is a version bump**, however small the diff —
   then `update_config_artifacts()`, then commit code, version, and hashes together.
   Import fails until you do.
7. **Configs must exist on disk before queue submission** — the queue stores paths only.
   That is exactly why `Blueprint` writes everything up front.
8. **Database failures must not be fatal.** Guard with `is_connected` / `use_database`
   and degrade gracefully.
9. **Refer to stages by `ProcessSpec.name`**, not by class name — `Photometry` serves
   three different stages.
10. **`is_pipeline=True` is for operational runs only.** Ad-hoc and user code should pass
    `False`; `pipeline_lock.py` enforces this for `DataReduction.run()`.
11. **Return code 2 is not a failure.** It means all inputs were sanity-rejected.
12. **The six `flag:` booleans in a `SciProcConfiguration` are the reprocessing API.**
    Flip them rather than deleting products; the `factory` directory makes reruns cheap.
13. **Never overwrite on unclear provenance.** `overwrite=True` can be escalated
    automatically from a stale or *missing* `runtime_version`. If the recorded-version
    bookkeeping is incomplete or self-contradictory, report it and stop — do not run
    anything that writes data.
14. **Stay in the repository you were given.** Production runs from a different checkout
    and environment; other users' spaces are off-limits unless you were asked. Treat the
    live volumes and databases as read-only during investigations.

---

## 7. Canonical usage

Off-pipeline coadd with custom settings — the canonical user entry point:

```python
from pipeline.config import SciProcConfiguration
from pipeline.imcoadd import ImCoadd

config = SciProcConfiguration.user_config(input_images)
config.node.imcoadd.coadd_image = "user_coadd.fits"
config.node.imcoadd.bkgsub_type = "constant"
ImCoadd(config).run()
```

Full operational run for a night:

```python
from pipeline.wrapper import DataReduction

dr = DataReduction(["2026-05-05"], use_db=True, is_pipeline=True)
dr.run(use_system_queue=True, input_type="Daily")
```

Path introspection — one instance answers everything about a group:

```python
from pipeline.path import PathHandler

p = PathHandler(raw_images, is_pipeline=True)
p.obj, p.filter, p.nightdate           # forwarded to NameHandler
p.processed_images                     # where preprocessing will write
p.preprocess.masterframe               # (mbias, mdark, mflat) for this group
p.astrometry.factory.catalog           # SExtractor catalog in the factory dir
p.imcoadd.coadd_image                  # final coadd path
p.imsubtract.diffim                    # difference image path
p.sciproc_output_yml, p.sciproc_output_log
```

Single-stage rerun from an existing config — stage names are `ProcessSpec.name`:

```python
from pipeline.run import run_scidata_reduction

run_scidata_reduction(
    "/lyman/data2/processed/2026-05-05/T02386/m650/T02386_m650_2026-05-05.yml",
    processes=["coadd", "coadd_photometry"],
    overwrite=True,
)
```

QA lookup and blast-radius analysis — ask the database, not the filesystem, and for
anything ad hoc just write the SQL:

```python
from pipeline.services.database import free_query

# Nightly QA for coadds
rows = free_query(
    "SELECT image_name, seeing, ul5_5, zp_auto FROM image_qa"
    " WHERE image_type = 'coadd' AND nightdate BETWEEN %s AND %s",
    ["2026-05-01", "2026-05-07"],
)

# Config-level progress for one night
status = free_query(
    "SELECT name, progress, status, errors, pipeline_version FROM process_status"
    " WHERE nightdate = %s AND config_type = 'science' ORDER BY name",
    ["2026-05-05"],
)

# Everything derived from a suspect master flat (recurse for the full cone)
contaminated = free_query(
    "SELECT d.derived_image_id, d.dependency_role, q.image_name"
    " FROM image_qa_dependency d JOIN image_qa q ON q.id = d.derived_image_id"
    " WHERE d.source_image_id = (SELECT id FROM image_qa WHERE image_name = %s)",
    ["flat_g_7DT01_2026-05-04_...fits"],
)
```

Reprocessing impact before you queue anything:

```python
from pipeline.services.database import ProcessStatusDependency

psd = ProcessStatusDependency()
psd.get_derived("2026-05-05_7DT01")      # configs that must rerun after this preproc
psd.get_sources("T02386_m650_2026-05-05")  # what this science config depends on
```

Recover a config whose per-stage state went stale:

```python
from pipeline.config import SciProcConfiguration
SciProcConfiguration.reset_config("/path/to/T02386_m650_2026-05-05.yml")
```

---

## 8. Repository layout

```
pipeline/
  path/        NameHandler, PathHandler, mixins, generators      ← backbone
  config/      BaseConfig, ConfigNode, Preproc/SciProc configs   ← backbone
  services/    blueprint, scheduler, queue, setup, checker,
               logger, monitor, memory, locks, database/         ← backbone
  const/       environ, observation, sciproc, preproc, run       ← backbone
  errors/      registry, definitions, composite exceptions       ← backbone
  utils/       collections, filesystem, header, tile, timing, config_integrity
  preprocess/ astrometry/ photometry/ imcoadd/ subtract/         ← science stages
  cuda/        optional CuPy kernels
  too/         ToO catalogs, plots, email, backfill
  tools/       analysis helpers for notebooks: quickvis, ds9 regions,
               catalog crossmatch, table filtering, PA statistics
  io/          FITS-LDAC reader/writer
  cli/         executable entry scripts invoked by the Scheduler
  run.py       run_preprocess, run_scidata_reduction, query_observations
  wrapper.py   DataReduction
ref/           base YAMLs, astromatic configs, QA criteria, hashes
systemd/       queue + trigger services and timers
run_examples/  runnable end-to-end examples
test/          ad-hoc scripts and benchmarks
```
