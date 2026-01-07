from __future__ import annotations

import types
from typing import Dict, Tuple, Type


class PipelineError(Exception):
    pass


# class PreprocessError(Exception):
#     """Errors that occurred during preprocess."""
#     pass

# class AstrometryError(Exception):
#     """Errors that occurred during astrometry."""

#     pass


# class PhotometryError(Exception):
#     """Errors that occurred during photometry."""

#     pass


"""
pipeline_errors.py

A small registry + factory that lets you refer to combined errors like:

    raise Astrometry.BadWcsSolutionError("WCS fit failed")

and convert back/forth between:
- combined exception classes/instances
- integer error codes (e.g. 203)

Encoding (customize if you want):
    combined_code = (process_code * 100) + kind_code
where process_code is 0..99 and kind_code is 0..99.

So Astrometry (2) + BadWcsSolutionError (3) => 2*100 + 3 = 203.
"""


from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type


# ---------------------------
# Core data + registry
# ---------------------------


@dataclass(frozen=True)
class ProcessInfo:
    name: str
    code: int


@dataclass(frozen=True)
class KindInfo:
    name: str
    code: int
    base_exc: Type[BaseException]


class ErrorRegistry:
    def __init__(self) -> None:
        self._process_by_name: Dict[str, ProcessInfo] = {}
        self._process_by_code: Dict[int, ProcessInfo] = {}
        self._kind_by_name: Dict[str, KindInfo] = {}
        self._kind_by_code: Dict[int, KindInfo] = {}

        # Cache of generated combined exception classes
        self._combo_class_cache: Dict[Tuple[int, int], Type[BaseException]] = {}

    # ---- registration ----

    def register_process(self, name: str, code: int) -> None:
        name = str(name)
        if not (0 <= code <= 99):
            raise ValueError(f"process code must be 0..99 (got {code})")
        if name in self._process_by_name:
            raise ValueError(f"process '{name}' already registered")
        if code in self._process_by_code:
            raise ValueError(f"process code {code} already registered")

        info = ProcessInfo(name=name, code=code)
        self._process_by_name[name] = info
        self._process_by_code[code] = info

    def register_kind(self, name: str, code: int, base_exc: Type[BaseException] = Exception) -> None:
        name = str(name)
        if not (0 <= code <= 99):
            raise ValueError(f"kind code must be 0..99 (got {code})")
        if name in self._kind_by_name:
            raise ValueError(f"kind '{name}' already registered")
        if code in self._kind_by_code:
            raise ValueError(f"kind code {code} already registered")

        info = KindInfo(name=name, code=code, base_exc=base_exc)
        self._kind_by_name[name] = info
        self._kind_by_code[code] = info

    # ---- lookup ----

    def process(self, name: str) -> ProcessInfo:
        try:
            return self._process_by_name[name]
        except KeyError:
            raise KeyError(f"unknown process '{name}'") from None

    def kind(self, name: str) -> KindInfo:
        try:
            return self._kind_by_name[name]
        except KeyError:
            raise KeyError(f"unknown kind '{name}'") from None

    def process_by_code(self, code: int) -> Optional[ProcessInfo]:
        return self._process_by_code.get(code)

    def kind_by_code(self, code: int) -> Optional[KindInfo]:
        return self._kind_by_code.get(code)

    # ---- encode/decode ----

    @staticmethod
    def encode(process_code: int, kind_code: int) -> int:
        if not (0 <= process_code <= 99):
            raise ValueError(f"process_code must be 0..99 (got {process_code})")
        if not (0 <= kind_code <= 99):
            raise ValueError(f"kind_code must be 0..99 (got {kind_code})")
        return process_code * 100 + kind_code

    @staticmethod
    def decode(error_code: int) -> Tuple[int, int]:
        if not (0 <= error_code <= 9999):
            raise ValueError(f"error_code must be 0..9999 (got {error_code})")
        return error_code // 100, error_code % 100


# ---------------------------
# Combined exception mechanics
# ---------------------------


class PipelineErrorMixin:
    """
    Mixin attached to every generated combined exception class.

    The combined exception class also subclasses the chosen base exception
    (e.g. ValueError, FileNotFoundError, etc) so isinstance checks still work.
    """

    process_name: str
    process_code: int
    kind_name: str
    kind_code: int
    error_code: int

    # Extra optional payload
    data: Dict[str, Any]

    def __init__(self, message: Optional[str] = None, *, cause: Optional[BaseException] = None, **data: Any) -> None:
        self.data = dict(data)
        if cause is not None:
            self.__cause__ = cause
        # Let the base exception store the message/args
        super().__init__(message if message is not None else "")  # type: ignore[misc]

    def __str__(self) -> str:
        base_msg = super().__str__()  # type: ignore[misc]
        prefix = f"[{self.error_code} {self.process_name}.{self.kind_name}]"
        return f"{prefix} {base_msg}".rstrip()


class UnknownProcessOrKindError(Exception):
    """Used when decoding an error code whose process/kind aren't registered."""


def _make_combo_class(
    *,
    registry: ErrorRegistry,
    process_info: ProcessInfo,
    kind_info: KindInfo,
) -> Type[BaseException]:
    code = registry.encode(process_info.code, kind_info.code)
    key = (process_info.code, kind_info.code)
    cached = registry._combo_class_cache.get(key)
    if cached is not None:
        return cached

    cls_name = f"{process_info.name}.{kind_info.name}"

    # Keep isinstance(e, ValueError) working by inheriting base_exc first.
    bases = (kind_info.base_exc, PipelineErrorMixin)

    attrs = {
        "__module__": __name__,
        "__qualname__": cls_name,
        "process_name": process_info.name,
        "process_code": process_info.code,
        "kind_name": kind_info.name,
        "kind_code": kind_info.code,
        "error_code": code,
    }

    combo_cls = type(cls_name, bases, attrs)  # type: ignore[arg-type]
    registry._combo_class_cache[key] = combo_cls
    return combo_cls


class ProcessNamespace:
    """
    Lets you write Astrometry.BadWcsSolutionError etc.
    """

    def __init__(self, registry: ErrorRegistry, process_name: str):
        self._registry = registry
        self._process = registry.process(process_name)

    @property
    def name(self) -> str:
        return self._process.name

    @property
    def code(self) -> int:
        return self._process.code

    def __getattr__(self, kind_name: str) -> Type[BaseException]:
        kind_info = self._registry.kind(kind_name)
        return _make_combo_class(registry=self._registry, process_info=self._process, kind_info=kind_info)

    def from_kind_code(self, kind_code: int) -> Type[BaseException]:
        kind_info = self._registry.kind_by_code(kind_code)
        if kind_info is None:
            raise KeyError(f"unknown kind_code {kind_code}")
        return _make_combo_class(registry=self._registry, process_info=self._process, kind_info=kind_info)


# ---------------------------
# Public helper functions
# ---------------------------


def error_code_of(exc: BaseException) -> Optional[int]:
    """
    If it's one of our combined exceptions, returns its integer error code.
    Otherwise returns None.
    """
    return getattr(exc, "error_code", None)


def exception_from_code(registry: ErrorRegistry, error_code: int, message: str = "") -> BaseException:
    """
    Decodes an integer error_code -> exception instance.
    If process/kind is unknown, raises UnknownProcessOrKindError.
    """
    p_code, k_code = registry.decode(error_code)
    p = registry.process_by_code(p_code)
    k = registry.kind_by_code(k_code)
    if p is None or k is None:
        raise UnknownProcessOrKindError(f"Unknown process/kind for code {error_code} (process={p_code}, kind={k_code})")
    cls = _make_combo_class(registry=registry, process_info=p, kind_info=k)
    return cls(message)


def code_from_names(registry: ErrorRegistry, process_name: str, kind_name: str) -> int:
    p = registry.process(process_name)
    k = registry.kind(kind_name)
    return registry.encode(p.code, k.code)


def names_from_code(registry: ErrorRegistry, error_code: int) -> Tuple[str, str]:
    p_code, k_code = registry.decode(error_code)
    p = registry.process_by_code(p_code)
    k = registry.kind_by_code(k_code)
    if p is None or k is None:
        raise UnknownProcessOrKindError(f"Unknown process/kind for code {error_code} (process={p_code}, kind={k_code})")
    return p.name, k.name


# ---------------------------
# Example setup (your mappings)
# ---------------------------

registry = ErrorRegistry()

# Processes (1..7) — you can add more (0..99)
registry.register_process("preprocess", 1)
registry.register_process("astrometry", 2)
registry.register_process("single_photometry", 3)
registry.register_process("combine", 4)
registry.register_process("combined_photometry", 5)
registry.register_process("subtraction", 6)
registry.register_process("difference_photometry", 7)

# Error kinds (0..99) — add as many as you want
registry.register_kind("ValueError", 1, ValueError)
registry.register_kind("FileNotFoundError", 2, FileNotFoundError)


# Your custom kind example: make it code 3 so Astrometry.BadWcsSolutionError -> 203
class BadWcsSolutionError(Exception):
    pass


registry.register_kind("BadWcsSolutionError", 3, BadWcsSolutionError)

# Namespaces you can import/use directly
PreprocessError = ProcessNamespace(registry, "preprocess")
AstrometryError = ProcessNamespace(registry, "astrometry")
SinglePhotometryError = ProcessNamespace(registry, "single_photometry")
CombineError = ProcessNamespace(registry, "combine")
CombinedPhotometryError = ProcessNamespace(registry, "combined_photometry")
SubtractionError = ProcessNamespace(registry, "subtraction")
DifferencePhotometryError = ProcessNamespace(registry, "difference_photometry")


# ---------------------------
# Quick self-demo (optional)
# ---------------------------
if __name__ == "__main__":
    # Create/raise a combined exception
    try:
        raise AstrometryError.BadWcsSolutionError("WCS fit failed", image_id="img_001")
    except Exception as e:
        print("Exception:", repr(e))
        print("String:", str(e))
        print("Is BadWcsSolutionError?", isinstance(e, BadWcsSolutionError))
        print("Is Exception?", isinstance(e, Exception))
        print("error_code_of:", error_code_of(e))  # 203
        print("data:", getattr(e, "data", None))

        # Convert back from code
        ex2 = exception_from_code(registry, 203, "decoded from code")
        print("Decoded:", str(ex2))

        # Name conversions
        print("names_from_code(203):", names_from_code(registry, 203))
        print(
            "code_from_names('astrometry','BadWcsSolutionError'):",
            code_from_names(registry, "astrometry", "BadWcsSolutionError"),
        )
