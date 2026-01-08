from __future__ import annotations


class PipelineError(Exception):
    pass


class UnknownProcessOrKindError(Exception):
    pass


# preprocess errors
class MasterFrameNotFoundError(Exception):
    pass


# astrometry errors
class BlankImageError(Exception):
    pass


class BadWcsSolutionError(Exception):
    pass


class PointingError(Exception):
    pass


class PositionAngleError(Exception):
    pass


# photometry errors
class NotEnoughSourcesError(Exception):
    pass


class NoReferenceSourceError(Exception):
    pass


# imstack errors
class BackgroundArtifactError(Exception):
    pass


class SeeingVariationError(Exception):
    pass


# subtraction errors
class NoReferenceImageError(Exception):
    pass


# generic errors
class QualityCheckFailedError(Exception):
    pass


class UnknownError(Exception):
    pass
