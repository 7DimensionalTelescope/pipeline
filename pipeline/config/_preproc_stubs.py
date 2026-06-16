# AUTO-GENERATED — do not edit manually.
# Source: preproc_base.yml
# Run update_config_artifacts() to regenerate.
from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.config.base import ConfigNode

    class InfoNode(ConfigNode):
        file: Any
        project: str
        creation_version: Any
        runtime_version: Any
        creation_datetime: Any
        last_update_datetime: Any

    class LoggingNode(ConfigNode):
        level: str
        file: Any
        format: str
        handlers: list

    class PreprocessNode(ConfigNode):
        masterframe: str
        max_offset: int
        n_sigma: int
        n_head_blocks: int
        use_multi_device: bool
        use_multi_thread: bool
        device: Any
        ignore_sanity_if_no_match: bool
        ignore_lenient_keys_if_no_match: bool

    class SettingsNode(ConfigNode):
        is_too: bool
        is_pipeline: bool

    class InputNode(ConfigNode):
        masterframe_images: Any
        science_images: Any
        grouped_raw_images: Any
        raw_dir: Any

    class PreprocNode(ConfigNode):
        name: Any
        process_id: Any
        info: InfoNode
        logging: LoggingNode
        preprocess: PreprocessNode
        settings: SettingsNode
        input: InputNode

