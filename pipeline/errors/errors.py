from __future__ import annotations


class PipelineError(Exception):
    pass


class UnknownProcessOrKindError(Exception):
    pass


# preprocess errors
class MasterFrameNotFoundError(Exception):
    pass


class SameNightMasterFrameNotFoundError(Exception):
    pass


# astrometry errors
class EarlyQARejectionError(Exception):
    """All images rejected by early QA"""

    pass


class BadWcsSolutionError(Exception):
    pass


class InvalidWcsSolutionError(Exception):
    pass


# class PointingError(Exception):
#     pass


# class PositionAngleError(Exception):
#     pass


class AlternativeSolverError(Exception):
    pass


class SolveFieldGenericError(Exception):
    pass


class ScampGenericError(Exception):
    """Note that there's ScampError too in the ProcessError registry.
    Mind namespace collisions"""

    pass


class AstrometryReferenceGenerationError(Exception):
    pass


class SolutionEvaluationFailedError(Exception):
    pass


# photometry errors
class NotEnoughSourcesError(Exception):
    pass


class NoReferenceSourceError(Exception):
    pass


class SextractorGenericError(Exception):
    pass


class FilterCheckError(Exception):
    pass


class FilterInventoryError(Exception):
    pass


class PhotometryReferenceGenerationError(Exception):
    pass


class InferredFilterMismatchError(Exception):
    pass


# imcoadd errors
class BackgroundArtifactError(Exception):
    pass


class SeeingVariationError(Exception):
    pass


class SwarpGenericError(Exception):
    pass


# subtraction errors
class ReferenceImageNotFoundError(Exception):
    pass


class HotpantsGenericError(Exception):
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


class PreviousStageError(Exception):
    pass


class EmptyInputError(Exception):
    pass


class UnknownError(Exception):
    pass


class ConnectionError(Exception):
    pass
