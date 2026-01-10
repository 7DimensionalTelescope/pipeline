from __future__ import annotations


class PipelineError(Exception):
    pass


class UnknownProcessOrKindError(Exception):
    pass


# preprocess errors
class MasterFrameNotFoundError(Exception):
    pass


class NoOnDateMasterFrameError(Exception):
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


class AlternativeSolverError(Exception):
    pass


class SolveFieldError(Exception):
    pass


class ScampError(Exception):
    pass


class AstrometryReferenceGenerationError(Exception):
    pass


# photometry errors
class NotEnoughSourcesError(Exception):
    pass


class NoReferenceSourceError(Exception):
    pass


class SextractorError(Exception):
    pass


class FilterCheckError(Exception):
    pass


class FilterInventoryError(Exception):
    pass


class PhotometryReferenceGenerationError(Exception):
    pass


# imcoadd errors
class BackgroundArtifactError(Exception):
    pass


class SeeingVariationError(Exception):
    pass


class SwarpError(Exception):
    pass


# subtraction errors
class ReferenceImageNotFoundError(Exception):
    pass


class HotpantsError(Exception):
    pass


# generic errors
class QualityCheckFailedError(Exception):
    pass


class AssumptionFailedError(Exception):
    pass


class PrerequisiteNotMetError(RuntimeError):
    pass


class GroupingError(Exception):
    """Input image inhomogeneities that lead to undefined behavior"""

    pass


class ParseError(ValueError):
    pass


class UnknownError(Exception):
    pass
