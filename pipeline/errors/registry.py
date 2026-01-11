from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

from .errors import UnknownProcessOrKindError


# Immutable records for names <-> codes
@dataclass(frozen=True)
class ProcessInfo:  # e.g., "astrometry", code=2
    name: str
    code: int


@dataclass(frozen=True)
class KindInfo:  # e.g., "BadWcsSolutionError", code=31, base_exc=BadWcsSolutionError
    name: str
    code: int
    base_exc: Type[BaseException]


class ErrorRegistry:
    """
    The code representation of a composite exception is for DB storage and querying.
    """

    def __init__(self) -> None:
        # Mapping Dicts
        self._process_by_name: Dict[str, ProcessInfo] = {}
        self._process_by_code: Dict[int, ProcessInfo] = {}
        self._kind_by_name: Dict[str, KindInfo] = {}
        self._kind_by_code: Dict[int, KindInfo] = {}

        # Caches
        self._process_base_cache: Dict[int, Type[BaseException]] = {}
        self._combo_class_cache: Dict[Tuple[int, int], Type[BaseException]] = {}

    def register_process(self, name: str, code: int) -> None:
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
        if not (0 <= code <= 99):
            raise ValueError(f"kind code must be 0..99 (got {code})")
        if name in self._kind_by_name:
            raise ValueError(f"kind '{name}' already registered")
        if code in self._kind_by_code:
            raise ValueError(f"kind code {code} already registered")
        info = KindInfo(name=name, code=code, base_exc=base_exc)
        self._kind_by_name[name] = info
        self._kind_by_code[code] = info

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

    @staticmethod
    def encode(process_code: int, kind_code: int) -> int:
        if not (0 <= process_code <= 99):
            raise ValueError(f"process_code must be 0..99 (got {process_code})")
        if not (0 <= kind_code <= 99):
            raise ValueError(f"kind_code must be 0..99 (got {kind_code})")
        return process_code * 100 + kind_code

    @staticmethod
    def decode(error_code: int) -> Tuple[int, int]:
        if not (0 <= error_code <= 999):
            raise ValueError(f"error_code must be 0..999 (got {error_code})")
        return error_code // 100, error_code % 100

    def exception_from_code(self, error_code: int, message: str = "", **data: Any) -> BaseException:
        """Create an exception instance from an error code using this registry."""
        cls = exception_class_from_code(self, error_code)
        return cls(message, **data)

    def names_from_code(self, error_code: int) -> Tuple[str, str]:
        """Get process and kind names from an error code using this registry."""
        p_code, k_code = self.decode(error_code)
        p = self.process_by_code(p_code)
        k = self.kind_by_code(k_code)
        if p is None or k is None:
            raise UnknownProcessOrKindError(
                f"Unknown process/kind for code {error_code} (process={p_code}, kind={k_code})"
            )
        return p.name, k.name


# ---------------------------
# Base mixin used by all generated exceptions
# ---------------------------


class PipelineErrorMixin:
    process_name: str
    process_code: int
    kind_name: str
    kind_code: int
    error_code: int
    data: Dict[str, Any]

    def __init__(self, message: Optional[str] = None, *, cause: Optional[BaseException] = None, **data: Any) -> None:
        self.data = dict(data)
        if cause is not None:
            self.__cause__ = cause
        super().__init__(message if message is not None else "")  # type: ignore[misc]

    def __str__(self) -> str:
        msg = super().__str__()  # type: ignore[misc]
        prefix = f"[{self.error_code} {self.process_name}.{self.kind_name}]"
        return f"{prefix} {msg}".rstrip()


# ---------------------------
# Process error class (metaclass makes `.ValueError` etc)
# ---------------------------


class ProcessErrorMeta(type):
    def __repr__(cls) -> str:
        return cls.__name__

    def __getattr__(cls, attr: str) -> Type[BaseException]:
        """
        Allows: AstrometryError.ValueError  (exact)
                AstrometryError.Value      (alias for ValueError)
                AstrometryError.BadWcsSolution (alias for BadWcsSolutionError)
        """
        reg: ErrorRegistry = getattr(cls, "_registry")
        pinfo: ProcessInfo = getattr(cls, "_process_info")

        # Direct match
        kinfo = reg._kind_by_name.get(attr)

        # Alias stripping/adding: try common "Error" suffix patterns
        if kinfo is None and not attr.endswith("Error"):
            kinfo = reg._kind_by_name.get(attr + "Error")
        if kinfo is None and attr.endswith("Error"):
            kinfo = reg._kind_by_name.get(attr[:-5])  # "FooError" -> "Foo"

        if kinfo is None:
            raise AttributeError(f"{cls.__name__} has no kind '{attr}'")

        return _make_combo_class(reg=reg, pinfo=pinfo, kinfo=kinfo)


class ProcessErrorBase(Exception, metaclass=ProcessErrorMeta):
    """
    Base type for all process-specific errors.
    You normally don't instantiate this directly; you raise/catch it.
    """

    _registry: ErrorRegistry
    _process_info: ProcessInfo

    # Default kind used when raising the process error itself:
    _default_kind_name: str = "ProcessError"

    process_name: str
    process_code: int
    kind_name: str
    kind_code: int
    error_code: int
    data: Dict[str, Any]

    def __init__(self, message: Optional[str] = None, *, cause: Optional[BaseException] = None, **data: Any) -> None:
        reg = self._registry
        pinfo = self._process_info
        kinfo = reg.kind(self._default_kind_name)

        # Stamp metadata like a normal combined error would
        self.process_name = pinfo.name
        self.process_code = pinfo.code
        self.kind_name = kinfo.name
        self.kind_code = kinfo.code
        self.error_code = reg.encode(pinfo.code, kinfo.code)

        self.data = dict(data)
        if cause is not None:
            self.__cause__ = cause

        super().__init__(message if message is not None else "")

    def __str__(self) -> str:
        """str(AstrometryError("test")) -> '[200 astrometry.ProcessError] test'"""
        msg = super().__str__()
        prefix = f"[{self.error_code} {self.process_name}.{self.kind_name}]"
        return f"{prefix} {msg}".rstrip()

    @classmethod
    def exception(cls, kind: Union[str, Type[BaseException], BaseException]) -> Type[BaseException]:
        """
        Resolve a 'kind' into a combined exception class under this process.
        It exists to do self.logger.process_error.exception(ValueError) in Photometry.

        Examples:
            AstrometryError.exception(ValueError) -> AstrometryError.ValueError
            AstrometryError.exception("ValueError") -> AstrometryError.ValueError
            AstrometryError.exception(ValueError("x")) -> AstrometryError.ValueError
            AstrometryError.exception(CoaddError.ValueError) -> CoaddError.ValueError
            AstrometryError.exception(AttributeError) -> AstrometryError.UnknownError
        """
        # If passed an instance, use its type
        if isinstance(kind, BaseException):
            kind_obj = type(kind)
        else:
            kind_obj = kind

        # If already one of our generated combined/process exception classes, return as-is
        if isinstance(kind_obj, type) and hasattr(kind_obj, "error_code"):
            return kind_obj  # type: ignore[return-value]

        # Determine the lookup name we will try on the process class
        if isinstance(kind_obj, str):
            name = kind_obj
        elif isinstance(kind_obj, type) and issubclass(kind_obj, BaseException):
            name = kind_obj.__name__
        else:
            raise TypeError(f"Unsupported kind: {kind!r}")

        # Try resolving to a registered kind under this process; otherwise fall back
        try:
            return getattr(cls, name)
        except AttributeError:
            # Must be registered as a kind name in your registry, e.g.
            # registry.register_kind("UnknownError", 99, UnknownError)
            return getattr(cls, "UnknownError")


class ComboErrorMeta(ProcessErrorMeta):
    """Pretty-print combo classes like 'AstrometryError.ValueError'."""

    def __repr__(cls) -> str:
        return cls.__qualname__

    # IMPORTANT: combo classes don't have _registry/_process_info, so disable the
    # kind-factory behavior inherited from ProcessErrorMeta.
    def __getattr__(cls, attr: str):
        raise AttributeError(attr)


def make_process_error(
    registry: ErrorRegistry, process_name: str, *, class_name: Optional[str] = None
) -> Type[ProcessErrorBase]:
    """
    Creates (or returns cached) process base exception class, e.g. AstrometryError.
    """
    pinfo = registry.process(process_name)
    cached = registry._process_base_cache.get(pinfo.code)
    if cached is not None:
        return cached  # type: ignore[return-value]

    cls_name = class_name or f"{process_name.title()}Error"

    default_kind = registry.kind(ProcessErrorBase._default_kind_name)

    attrs = {
        "__module__": __name__,
        "_registry": registry,
        "_process_info": pinfo,
        # nice-to-have class metadata (so AstrometryError.error_code works)
        "process_name": pinfo.name,
        "process_code": pinfo.code,
        "kind_name": default_kind.name,
        "kind_code": default_kind.code,
        "error_code": registry.encode(pinfo.code, default_kind.code),
    }

    proc_cls = type(cls_name, (ProcessErrorBase,), attrs)
    registry._process_base_cache[pinfo.code] = proc_cls
    return proc_cls


def _make_combo_class(*, reg: ErrorRegistry, pinfo: ProcessInfo, kinfo: KindInfo) -> Type[BaseException]:
    key = (pinfo.code, kinfo.code)
    cached = reg._combo_class_cache.get(key)
    if cached is not None:
        return cached

    proc_base = reg._process_base_cache.get(pinfo.code)
    if proc_base is None:
        # If user didn't create the process class yet, make a default one.
        proc_base = make_process_error(reg, pinfo.name, class_name=f"{pinfo.name.title()}Error")

    code = reg.encode(pinfo.code, kinfo.code)
    cls_name = f"{proc_base.__name__}.{kinfo.name}"

    # Inherit kind base first so isinstance(e, ValueError) works,
    # then process base so you can catch except AstrometryError: ...
    bases = (kinfo.base_exc, proc_base, PipelineErrorMixin)

    attrs = {
        "__module__": __name__,
        "__qualname__": cls_name,
        "process_name": pinfo.name,
        "process_code": pinfo.code,
        "kind_name": kinfo.name,
        "kind_code": kinfo.code,
        "error_code": code,
    }

    combo_cls = ComboErrorMeta(cls_name, bases, attrs)
    reg._combo_class_cache[key] = combo_cls
    return combo_cls


def exception_class_from_code(registry: ErrorRegistry, error_code: int) -> Type[BaseException]:
    p_code, k_code = registry.decode(error_code)
    p = registry.process_by_code(p_code)
    k = registry.kind_by_code(k_code)
    if p is None or k is None:
        raise UnknownProcessOrKindError(f"Unknown process/kind for code {error_code} (process={p_code}, kind={k_code})")
    return _make_combo_class(reg=registry, pinfo=p, kinfo=k)


ExceptionArg = Union[
    None,
    str,  # "ValueError" / "BadWcsSolution" / "BadWcsSolutionError"
    Type[BaseException],  # ValueError / BadWcsSolutionError / AstrometryError.ValueError
    BaseException,  # ValueError("x") / AstrometryError.ValueError("x")
]
