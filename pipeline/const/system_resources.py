# Computing Hardware
NUM_GPUS = 2  # maximum number of GPUs to use
MAX_WORKERS = 128
LOCAL_FSTYPES = ("ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs", "reiserfs")

# Multi-epoch scratch budget (services.utils.scratch_budget)
SCRATCH_SAFETY = 0.9  # fraction of free + deletable bytes a run may plan on
SCRATCH_RESERVE_GB = 200  # never planned on: a full scratch disk is an outage
SCRATCH_IDLE_HOURS = 6.0  # a factory tree untouched this long is deletable
SCRATCH_PROBE_SECONDS = 30  # claim-time scratch walk is reused this long
FACTORY_PLANES = 3  # SWarp-grid planes per input in a multi-epoch factory: bkgsub, resampled sci + wht
STAGE_PLANES = 2  # SWarp-grid planes per input staged for the combine: resampled sci + wht
