"""Typed failures exposed by the annotation module interface."""


class AnnotationError(Exception):
    """Base class for caller-actionable annotation failures."""


class ValidationError(AnnotationError):
    pass


class AuthorizationError(AnnotationError):
    pass


class ConflictError(AnnotationError):
    pass


class NotFoundError(AnnotationError):
    pass
