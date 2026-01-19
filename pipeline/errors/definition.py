from typing import Any, Tuple

from .registry import ErrorRegistry, make_process_error
from .errors import *

# ---------------------------
# Setup
# ---------------------------

registry = ErrorRegistry()

# =============================================================================
#                             Process Errors
# =============================================================================

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

# external program wrapper errors
# registry.register_process("sextractor", 20)
registry.register_process("solve-field", 21)
registry.register_process("scamp", 22)
# registry.register_process("swarp", 23)
# registry.register_process("hotpants", 24)


# Future placeholders. Uncomment when the need arises.
# registry.register_process("db", 20)  # You may need separate SQLiteDBError, PostgresDBError, etc.
# registry.register_process("main_db", 21)  # Main PostgresDB image db
# registry.register_kkocess("queue", 30)


# =============================================================================
#                               Kind Errors
# =============================================================================

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
registry.register_kind("ConnectionError", 12, ConnectionError)

# Pipeline specific errors
# CAVEAT: these are dynamically created and not understood by IDEs.
#         e.g., Rename Symbol won't work exhaustively. Use grep -r --binary-files=without-match
# preprocess from 20
registry.register_kind("MasterFrameNotFoundError", 20, MasterFrameNotFoundError)
registry.register_kind("SameNightMasterFrameNotFoundError", 21, SameNightMasterFrameNotFoundError)


# astrometry from 30
registry.register_kind("EarlyQARejectionError", 30, EarlyQARejectionError)
registry.register_kind("BadWcsSolutionError", 31, BadWcsSolutionError)
registry.register_kind("InvalidWcsSolutionError", 32, InvalidWcsSolutionError)
# registry.register_kind("PointingError", 32, PointingError)
# registry.register_kind("PositionAngleError", 33, PositionAngleError)
registry.register_kind("AlternativeSolverError", 34, AlternativeSolverError)
registry.register_kind("SolveFieldGenericError", 35, SolveFieldGenericError)
registry.register_kind("ScampGenericError", 36, ScampGenericError)
registry.register_kind("AstrometryReferenceGenerationError", 37, AstrometryReferenceGenerationError)
registry.register_kind("SolutionEvaluationFailedError", 38, SolutionEvaluationFailedError)

# photometry from 50
registry.register_kind("NotEnoughSourcesError", 50, NotEnoughSourcesError)
registry.register_kind("NoReferenceSourceError", 51, NoReferenceSourceError)
registry.register_kind("SextractorError", 52, SextractorGenericError)
registry.register_kind("FilterCheckError", 53, FilterCheckError)
registry.register_kind("FilterInventoryError", 54, FilterInventoryError)
registry.register_kind("PhotometryReferenceGenerationError", 57, PhotometryReferenceGenerationError)
registry.register_kind("InferredFilterMismatchError", 58, InferredFilterMismatchError)


# imcoadd from 60
registry.register_kind("BackgroundArtifactError", 60, BackgroundArtifactError)
registry.register_kind("SeeingVariationError", 61, SeeingVariationError)
# registry.register_kind("SwarpGenericError", 62, SwarpGenericError)


# subtraction from 70
registry.register_kind("ReferenceImageNotFoundError", 70, ReferenceImageNotFoundError)
# registry.register_kind("HotpantsGenericError", 71, HotpantsGenericError)


# generic from 80
registry.register_kind("QualityCheckFailedError", 80, QualityCheckFailedError)
registry.register_kind("AssumptionFailedError", 81, AssumptionFailedError)
registry.register_kind("PrerequisiteNotMetError", 82, PrerequisiteNotMetError)
registry.register_kind("GroupingError", 83, GroupingError)
registry.register_kind("ParseError", 84, ParseError)
registry.register_kind("PreviousStageError", 85, PreviousStageError)
registry.register_kind("EmptyInputError", 86, EmptyInputError)


# Reserved sentinel
registry.register_kind("UnknownError", 99, UnknownError)


# Create process exception classes
PreprocessError = make_process_error(registry, "preprocess", class_name="PreprocessError")
AstrometryError = make_process_error(registry, "astrometry", class_name="AstrometryError")
# generic photometry errors default to SinglePhotometryError
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
# MainDatabaseError = make_process_error(registry, "main_db", class_name="MainDatabaseError")
SolveFieldError = make_process_error(registry, "solve-field", class_name="SolveFieldError")
ScampError = make_process_error(registry, "scamp", class_name="ScampError")

# ------------------------------------------------------
# Convenience function using default registry
# ------------------------------------------------------


def exception_from_code(error_code: int, message: str = "", **data: Any) -> BaseException:
    """Create an exception instance from an error code using the default registry."""
    return registry.exception_from_code(error_code, message, **data)


def names_from_code(error_code: int) -> Tuple[str, str]:
    """Get process and kind names from an error code using the default registry."""
    return registry.names_from_code(error_code)
