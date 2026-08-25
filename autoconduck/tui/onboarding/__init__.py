from .helpers import *
from .screens import (
    _TEXTUAL,
    OnboardingScreen,
    ModelSourceScreen,
    ModelSelectionScreen,
)
from .screens_custom import ApiKeyScreen, CustomProvidersScreen
from .screens_extra import ProviderFormScreen, LauncherIntegrationScreen
from .screens_slm import SLMSetupScreen
from .screens_models import ModelCatalogScreen
from .slm_progress import DownloadProgress, download_progress
from .health import build_health_matrix, probe_connectivity, render_health_matrix


def _require_textual() -> None:
    """Raise a useful error when a screen is used without Textual installed."""
    if not _TEXTUAL:
        raise RuntimeError("Textual is required for the interactive TUI")

__all__ = [
    "AGENTS",
    "format_price",
    "render_agent_rows",
    "render_source_rows",
    "render_provider_rows",
    "render_slm_rows",
    "move_cursor",
    "detect_agents",
    "is_agent_configured",
    "MODELS_PLACEHOLDER",
    "OnboardingScreen",
    "ModelSourceScreen",
    "ModelSelectionScreen",
    "ApiKeyScreen",
    "CustomProvidersScreen",
    "ProviderFormScreen",
    "LauncherIntegrationScreen",
    "SLMSetupScreen",
    "ModelCatalogScreen",
    "DownloadProgress",
    "download_progress",
    "build_health_matrix",
    "probe_connectivity",
    "render_health_matrix",
]
