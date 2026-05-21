"""Backend implementations for local and optional remote execution."""

from .base import Backend, ModelHandle
from .local import LocalTransformersBackend
from .openweights import OpenWeightsBackend

__all__ = ["Backend", "LocalTransformersBackend", "ModelHandle", "OpenWeightsBackend"]
