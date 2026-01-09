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
registry.register_process("coadded_photometry", 5)
registry.register_process("subtraction", 6)
registry.register_process("difference_photometry", 7)
# errors outside specific processes (orchestrator, config, PathHandler, user-input, etc.)
registry.register_process("system", 9)
registry.register_process("undefined_process", 0)


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

# Other candidates; revise the error code to use them

# # --- Common programming / runtime errors ---
# registry.register_kind("TypeError", 3, TypeError)
# registry.register_kind("KeyError", 4, KeyError)
# registry.register_kind("IndexError", 5, IndexError)
# registry.register_kind("AttributeError", 6, AttributeError)
# registry.register_kind("AssertionError", 7, AssertionError)
# registry.register_kind("NameError", 8, NameError)
# registry.register_kind("UnboundLocalError", 9, UnboundLocalError)  # NameError subtype
# registry.register_kind("RuntimeError", 10, RuntimeError)
# registry.register_kind("NotImplementedError", 11, NotImplementedError)
# registry.register_kind("RecursionError", 12, RecursionError)
# registry.register_kind("EOFError", 13, EOFError)

# # --- Import / module loading ---
# registry.register_kind("ImportError", 14, ImportError)
# registry.register_kind("ModuleNotFoundError", 15, ModuleNotFoundError)

# # --- Syntax / parsing (rare in production runtime, but “major”) ---
# registry.register_kind("SyntaxError", 16, SyntaxError)
# registry.register_kind("IndentationError", 17, IndentationError)
# registry.register_kind("TabError", 18, TabError)

# # --- Iteration / generators (useful if you surface control-flow errors) ---
# registry.register_kind("StopIteration", 19, StopIteration)
# registry.register_kind("StopAsyncIteration", 20, StopAsyncIteration)
# registry.register_kind("GeneratorExit", 21, GeneratorExit)  # BaseException subclass

# # --- Arithmetic / numeric ---
# registry.register_kind("ZeroDivisionError", 22, ZeroDivisionError)
# registry.register_kind("OverflowError", 23, OverflowError)
# registry.register_kind("FloatingPointError", 24, FloatingPointError)

# # --- OS / IO family (OSError + common subclasses) ---
# registry.register_kind("OSError", 30, OSError)
# registry.register_kind("PermissionError", 31, PermissionError)
# registry.register_kind("FileExistsError", 32, FileExistsError)
# registry.register_kind("IsADirectoryError", 33, IsADirectoryError)
# registry.register_kind("NotADirectoryError", 34, NotADirectoryError)
# registry.register_kind("InterruptedError", 35, InterruptedError)
# registry.register_kind("TimeoutError", 36, TimeoutError)
# registry.register_kind("BlockingIOError", 37, BlockingIOError)

# # --- Networking / pipes (all are OSError subclasses) ---
# registry.register_kind("ConnectionError", 40, ConnectionError)
# registry.register_kind("ConnectionRefusedError", 41, ConnectionRefusedError)
# registry.register_kind("ConnectionResetError", 42, ConnectionResetError)
# registry.register_kind("ConnectionAbortedError", 43, ConnectionAbortedError)
# registry.register_kind("BrokenPipeError", 44, BrokenPipeError)

# # --- Process / IPC ---
# registry.register_kind("ChildProcessError", 45, ChildProcessError)
# registry.register_kind("ProcessLookupError", 46, ProcessLookupError)

# # --- Resources / buffers / memory ---
# registry.register_kind("MemoryError", 50, MemoryError)
# registry.register_kind("BufferError", 51, BufferError)

# # --- Unicode / encoding ---
# registry.register_kind("UnicodeError", 60, UnicodeError)
# registry.register_kind("UnicodeDecodeError", 61, UnicodeDecodeError)
# registry.register_kind("UnicodeEncodeError", 62, UnicodeEncodeError)
# registry.register_kind("UnicodeTranslateError", 63, UnicodeTranslateError)

# # --- Exception groups (Python 3.11+) ---
# registry.register_kind("ExceptionGroup", 70, ExceptionGroup)
# registry.register_kind("BaseExceptionGroup", 71, BaseExceptionGroup)

# # --- “System-ish” BaseException types (only register if you want to encode them) ---
# registry.register_kind("KeyboardInterrupt", 80, KeyboardInterrupt)
# registry.register_kind("SystemExit", 81, SystemExit)


# Pipeline specific errors
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
registry.register_kind("SwarpError", 62, SwarpError)


# subtraction from 70
registry.register_kind("NoReferenceImageError", 70, NoReferenceImageError)
registry.register_kind("HotpantsError", 71, HotpantsError)


# generic from 80
registry.register_kind("QualityCheckFailedError", 80, QualityCheckFailedError)
registry.register_kind("AssumptionFailedError", 81, AssumptionFailedError)
registry.register_kind("PrerequisiteNotMetError", 82, PrerequisiteNotMetError)


# Reserved sentinel
registry.register_kind("UnknownError", 99, UnknownError)


# Create process exception classes
PreprocessError = make_process_error(registry, "preprocess", class_name="PreprocessError")
AstrometryError = make_process_error(registry, "astrometry", class_name="AstrometryError")
SinglePhotometryError = make_process_error(registry, "single_photometry", class_name="SinglePhotometryError")
CoaddError = make_process_error(registry, "coadd", class_name="CoaddError")
CoaddedPhotometryError = make_process_error(registry, "coadded_photometry", class_name="CoaddedPhotometryError")
SubtractionError = make_process_error(registry, "subtraction", class_name="SubtractionError")
DifferencePhotometryError = make_process_error(registry, "difference_photometry", class_name="DifferencePhotometryError")  # fmt: skip
# Non-process-specific exception classes
UndefinedProcessError = make_process_error(registry, "undefined_process", class_name="UndefinedProcessError")
SystemError = make_process_error(registry, "system", class_name="SystemError")


# ------------------------------------------------------
# Convenience function using default registry
# ------------------------------------------------------


def exception_from_code(error_code: int, message: str = "", **data: Any) -> BaseException:
    """Create an exception instance from an error code using the default registry."""
    return registry.exception_from_code(error_code, message, **data)


def names_from_code(error_code: int) -> Tuple[str, str]:
    """Get process and kind names from an error code using the default registry."""
    return registry.names_from_code(error_code)
