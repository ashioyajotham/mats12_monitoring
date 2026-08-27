"""Generation-provider adapters with explicit audit contracts."""

from src.backends.zai import ZAIBackend, ZAIBackendError

__all__ = ["ZAIBackend", "ZAIBackendError"]
