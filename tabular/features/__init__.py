# ml_foundry/features/__init__.py

# Clean module initialization - registration is handled by the SearchPathPlugin
# This keeps the package clean and follows Hydra best practices

from .base import FeatureGenerator

__all__ = ["FeatureGenerator"]