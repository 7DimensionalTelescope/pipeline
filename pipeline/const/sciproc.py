from __future__ import annotations

from dataclasses import dataclass

REJECTION_PROCESS_HEADER_KEY = "REJ_PROC"


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    """Canonical description of one science-process step."""

    name: str
    error_code: int
    progress_start: int
    progress_end: int
    yml_key: str  # this must be in sync with sciproc_base.yml
    photometry_mode: str | None = None
    milestones: tuple[tuple[str, int], ...] = ()


class SciProcessRegistry:
    """Single source of truth for science-process names, ordering, and progress."""

    def __init__(self, processes: list[ProcessSpec]):
        self._processes = tuple(sorted(processes, key=lambda process: process.progress_start))
        self._by_name = {process.name: process for process in self._processes}

        self._validate_unique("name", [process.name for process in self._processes])
        self._validate_unique("code", [process.error_code for process in self._processes])
        self._validate_unique("progress", [process.progress_start for process in self._processes])
        self._validate_unique("yml_key", [process.yml_key for process in self._processes])
        for process in self._processes:
            if process.progress_end < process.progress_start:
                raise ValueError(
                    f"Invalid progress range for {process.name}: {process.progress_start} -> {process.progress_end}"
                )
            milestone_names = [name for name, _ in process.milestones]
            self._validate_unique(f"{process.name} milestones", milestone_names)
            for milestone_name, milestone_progress in process.milestones:
                if not (process.progress_start <= milestone_progress <= process.progress_end):
                    raise ValueError(
                        f"Invalid milestone for {process.name}: {milestone_name}={milestone_progress} "
                        f"outside {process.progress_start}..{process.progress_end}"
                    )

    @staticmethod
    def _validate_unique(field_name: str, values: list[object]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate process registry {field_name} values: {values}")

    @property
    def specs(self) -> tuple[ProcessSpec, ...]:
        return self._processes

    def get(self, name: str) -> ProcessSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown process name: {name}") from exc

    def configured_progress(self, process: str | ProcessSpec) -> int:
        process_info = self.get(process) if isinstance(process, str) else process
        return process_info.progress_start

    def completed_progress(self, process: str | ProcessSpec) -> int:
        process_info = self.get(process) if isinstance(process, str) else process
        return process_info.progress_end

    def milestone_progress(self, process: str | ProcessSpec, milestone: str) -> int:
        process_info = self.get(process) if isinstance(process, str) else process
        for milestone_name, milestone_progress in process_info.milestones:
            if milestone_name == milestone:
                return milestone_progress
        raise KeyError(f"Unknown milestone for {process_info.name}: {milestone}")

    def step_progress(self, process: str | ProcessSpec, step: int, total_steps: int) -> int:
        process_info = self.get(process) if isinstance(process, str) else process
        start = self.configured_progress(process_info)
        end = self.completed_progress(process_info)

        if total_steps <= 0:
            return end

        clamped_step = min(max(step, 0), total_steps)
        progress = start + (end - start) * (clamped_step / total_steps)
        return int(round(progress))


SCIPROC_PROCESSES = [
    ProcessSpec(
        name="astrometry",
        error_code=2,
        progress_start=0,
        progress_end=20,
        yml_key="astrometry",
        milestones=(
            ("scamp_prep", 5),
            ("all_rejected_early_qa", 5),
            ("scamp_main", 10),
            ("scamp_main_eval", 15),
        ),
    ),
    ProcessSpec(
        name="single_photometry",
        error_code=3,
        progress_start=40,
        progress_end=60,
        yml_key="single_photometry",
        photometry_mode="single_photometry",
    ),
    ProcessSpec(
        name="coadd",
        error_code=4,
        progress_start=60,
        progress_end=70,
        yml_key="coadd",
        milestones=(
            ("bkgsub", 61),
            ("zpscale", 62),
            ("calculate_weight_map", 63),
            ("apply_bpmask", 64),
            ("joint_registration", 65),
            ("run_convolution", 66),
            ("coadd_with_swarp", 68),
            ("plot_coadd_image", 69),
        ),
    ),
    ProcessSpec(
        name="coadd_photometry",
        error_code=5,
        progress_start=70,
        progress_end=80,
        yml_key="coadd_photometry",
        photometry_mode="coadd_photometry",
    ),
    ProcessSpec(
        name="subtraction",
        error_code=6,
        progress_start=80,
        progress_end=90,
        yml_key="subtraction",
        milestones=(
            ("define_paths", 82),
            ("create_substamps", 84),
            ("create_masks", 86),
            ("run_hotpants", 88),
        ),
    ),
    ProcessSpec(
        name="difference_photometry",
        error_code=7,
        progress_start=90,
        progress_end=100,
        yml_key="difference_photometry",
        photometry_mode="difference_photometry",
    ),
]

SCIPROCESS_REGISTRY = SciProcessRegistry(SCIPROC_PROCESSES)
