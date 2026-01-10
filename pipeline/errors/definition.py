from typing import Any, Tuple

from .registry import ErrorRegistry, make_process_error
from .errors import *

# ---------------------------
# Setup
# ---------------------------

registry = ErrorRegistry()


# Processes (1..7) - Specific pipeline processes
registry.register_process("preprocess", 1)
registry.register_process("astrometry", 2)
registry.register_process("single_photometry", 3)
registry.register_process("coadd", 4)
registry.register_process("coadd_photometry", 5)
registry.register_process("subtraction", 6)
registry.register_process("difference_photometry", 7)
# errors outside specific processes (orchestrator, config, PathHandler, user-input, etc.)
registry.register_process("system", 9)  # errors from outside scientific processes
registry.register_process("undefined_process", 0)

# system errors, but important enough to get their own names
registry.register_process("pathhandler", 11)
registry.register_process("config", 12)
registry.register_process("logger", 13)

# Future placeholders. Uncomment when the need arises.
# registry.register_process("db", 20)  # You may need separate SQLiteDBError, PostgresDBError, etc.
# registry.register_process("queue", 30)

# Error kinds (0..99) -- add new when needed
registry.register_kind("ProcessError", 0, Exception)  # for raising ProcessError itself like AstrometryError
registry.register_kind("ValueError", 1, ValueError)
registry.register_kind("TypeError", 2, TypeError)
registry.register_kind("IndexError", 3, IndexError)
registry.register_kind("KeyError", 4, KeyError)
registry.register_kind("FileNotFoundError", 6, FileNotFoundError)
registry.register_kind("PermissionError", 7, PermissionError)
registry.register_kind("TimeoutError", 8, TimeoutError)
registry.register_kind("KeyboardInterrupt", 9, KeyboardInterrupt)
registry.register_kind("NotImplementedError", 11, NotImplementedError)


# Pipeline specific errors
# CAVEAT: these are dynamically created and not understood by IDEs. e.g., Rename Symbol won't work exhaustively.
# preprocess from 20
registry.register_kind("MasterFrameNotFoundError", 20, MasterFrameNotFoundError)
registry.register_kind("NoOnDateMasterFrameError", 21, NoOnDateMasterFrameError)


# astrometry from 30
registry.register_kind("BlankImageError", 30, BlankImageError)
registry.register_kind("BadWcsSolutionError", 31, BadWcsSolutionError)
registry.register_kind("PointingError", 32, PointingError)
registry.register_kind("PositionAngleError", 33, PositionAngleError)
registry.register_kind("AlternativeSolverError", 34, AlternativeSolverError)
registry.register_kind("SolveFieldError", 35, SolveFieldError)
registry.register_kind("ScampError", 36, ScampError)
registry.register_kind("AstrometryReferenceGenerationError", 37, AstrometryReferenceGenerationError)


# photometry from 50
registry.register_kind("NotEnoughSourcesError", 50, NotEnoughSourcesError)
registry.register_kind("NoReferenceSourceError", 51, NoReferenceSourceError)
registry.register_kind("SextractorError", 52, SextractorError)
registry.register_kind("FilterCheckError", 53, FilterCheckError)
registry.register_kind("FilterInventoryError", 54, FilterInventoryError)
registry.register_kind("PhotometryReferenceGenerationError", 57, PhotometryReferenceGenerationError)


# imcoadd from 60
registry.register_kind("BackgroundArtifactError", 60, BackgroundArtifactError)
registry.register_kind("SeeingVariationError", 61, SeeingVariationError)
# registry.register_kind("SwarpError", 62, SwarpError)  # maybe SwarpTimeoutError?


# subtraction from 70
registry.register_kind("ReferenceImageNotFoundError", 70, ReferenceImageNotFoundError)
# registry.register_kind("HotpantsError", 71, HotpantsError)


# generic from 80
registry.register_kind("QualityCheckFailedError", 80, QualityCheckFailedError)
registry.register_kind("AssumptionFailedError", 81, AssumptionFailedError)
registry.register_kind("PrerequisiteNotMetError", 82, PrerequisiteNotMetError)
registry.register_kind("GroupingError", 83, GroupingError)


# Reserved sentinel
registry.register_kind("UnknownError", 99, UnknownError)


# Create process exception classes
PreprocessError = make_process_error(registry, "preprocess", class_name="PreprocessError")
AstrometryError = make_process_error(registry, "astrometry", class_name="AstrometryError")
SinglePhotometryError = make_process_error(registry, "single_photometry", class_name="SinglePhotometryError")
CoaddError = make_process_error(registry, "coadd", class_name="CoaddError")
CoaddPhotometryError = make_process_error(registry, "coadd_photometry", class_name="CoaddPhotometryError")
SubtractionError = make_process_error(registry, "subtraction", class_name="SubtractionError")
DifferencePhotometryError = make_process_error(registry, "difference_photometry", class_name="DifferencePhotometryError")  # fmt: skip
# Non-process-specific exception classes
UndefinedProcessError = make_process_error(registry, "undefined_process", class_name="UndefinedProcessError")
SystemError = make_process_error(registry, "system", class_name="SystemError")
PathHandlerError = make_process_error(registry, "pathhandler", class_name="PathHandlerError")
ConfigurationError = make_process_error(registry, "config", class_name="ConfigurationError")
LoggerError = make_process_error(registry, "logger", class_name="LoggerError")


# ------------------------------------------------------
# Convenience function using default registry
# ------------------------------------------------------


def exception_from_code(error_code: int, message: str = "", **data: Any) -> BaseException:
    """Create an exception instance from an error code using the default registry."""
    return registry.exception_from_code(error_code, message, **data)


def names_from_code(error_code: int) -> Tuple[str, str]:
    """Get process and kind names from an error code using the default registry."""
    return registry.names_from_code(error_code)
