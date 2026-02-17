"""Custom exceptions for alabwebgazer."""

from __future__ import annotations


class AlabWebGazerError(Exception):
    """Base exception for this package."""


class ConfigError(AlabWebGazerError):
    """Raised when configuration is invalid."""


class DataValidationError(AlabWebGazerError):
    """Raised when input data violate required schema/constraints."""


class PipelineError(AlabWebGazerError):
    """Raised when a pipeline step fails in a non-validation way."""
