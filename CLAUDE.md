# CLAUDE.md

Instructions for AI agents working in this repository (7DT optical image reduction
pipeline). These override default behavior.

## 1. Read the codebase memory first

`.claude/memory/` is the machine-readable memory of this codebase. Before any
non-trivial task:

- Read `.claude/memory/INDEX.md` (map + task-to-file table) and
  `.claude/memory/invariants.md` (never-do rules) in full.
- Load the per-subsystem files your task touches (INDEX.md tells you which).

**Maintenance duty:** the memory is living documentation. When your change affects
something a memory file documents, update that file in the same piece of work and set
its `last_verified` to the new commit. If you find the memory contradicts the code,
fix the memory.

## 2. Hard rules (apply even if you read nothing else)

- **Never scan the filesystem to find data.** The data tree spans several NFS mounts
  and only raw frames are canonical. Ask `PathHandler` or the databases
  (`RawFrameQuery`; `free_query` for ad-hoc SQL — `RawImageQuery` is deprecated). See
  invariants.md §1 for the sanctioned exceptions and the raw-DB anchoring policy.
- **Never run anything that writes or overwrites data when recorded-version
  bookkeeping is inconsistent** (missing/stale `runtime_version` in a config, flags
  disagreeing with products on disk). `overwrite=True` is escalated automatically from
  stale or missing `runtime_version` (`services/version_check.py`). Report what you
  found and stop.
- **PathHandler / NameHandler are the sole authority** for FITS paths, filenames, and
  type parsing. Never write another normalizer; extend them instead.
- **Never edit files under `ref/` casually** — they are hash-locked; an unaccompanied
  edit makes `import pipeline` raise for everyone. Follow the version-bump ritual in
  `.claude/memory/conventions.md`.
- **Production runs from the stable clone** `/home/pipeline-stable/pipeline`, but this
  working tree shares the live Postgres, sqlite DBs, and NFS data. Treat DB writes and
  data-dir writes as production actions.

## 3. Comments and docstrings

- Never remove existing comments, commented-out code blocks, or stale docstrings. If
  they look wrong or stale, report and preserve them as-is.
- Function docstrings: one short line of keywords; Python type hints carry the detail.
- Explanatory comments inside code: one-liner or none — clean code is self-explanatory.
  A one-line section-header comment is fine. Mimic the existing comment style.

## 4. Working style

- **Think before coding.** State assumptions; if multiple interpretations exist,
  present them instead of picking silently; push back when a simpler approach exists;
  if something is unclear, stop and ask.
- **Simplicity first.** Minimum code that solves the problem: no speculative features,
  abstractions for single-use code, or unrequested configurability.
- **Surgical changes.** Touch only what the task requires; match existing style; don't
  refactor or "improve" adjacent code. Remove only orphans your own change created.
  Every changed line should trace to the request.
- **Verify against a goal.** Turn tasks into checkable success criteria and loop until
  they pass. There is no test suite (`test/` is not one — see conventions.md), so state
  how you verified: a targeted script, a dry run, an import check, a log inspection.
