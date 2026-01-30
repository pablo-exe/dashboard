"""
Configuration Module for RAG Dashboard
======================================

Centralized configuration management with support for:
- Environment variables
- Feature flags
- Default values

All configuration is read-only and loaded at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TraceViewerConfig:
    """Configuration for the LLM Trace Viewer component."""
    
    # Whether the trace viewer is enabled at all
    enabled: bool = True
    
    # Maximum JSON size (bytes) to render without virtualization warning
    max_json_size_warning: int = 1_000_000  # 1MB
    
    # Whether to show performance metrics
    show_token_usage: bool = True
    
    # Default tab to show ("trace" or "json")
    default_tab: str = "trace"


@dataclass(frozen=True)
class DashboardConfig:
    """Main dashboard configuration."""
    
    # Database settings
    experiments_db_path: Optional[str] = None
    onedrive_db_url: Optional[str] = None
    always_refresh_db: bool = False
    
    # Feature flags
    trace_viewer: TraceViewerConfig = None
    
    def __post_init__(self):
        # Handle mutable default
        if self.trace_viewer is None:
            object.__setattr__(self, 'trace_viewer', TraceViewerConfig())


def _parse_bool(value: str, default: bool = False) -> bool:
    """Parse a boolean from environment variable."""
    if not value:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def load_config() -> DashboardConfig:
    """
    Load configuration from environment variables.
    
    Environment Variables:
    ----------------------
    EXPERIMENTS_DB_PATH: Path to the DuckDB database file
    ONEDRIVE_DB_URL: URL to download the database from OneDrive
    ALWAYS_REFRESH_DB: Whether to always refresh the database on load
    
    TRACE_VIEWER_ENABLED: Enable/disable the trace viewer (default: true)
    REASONING_REVEAL_ALLOWED: Allow revealing raw reasoning (default: true)
    TRACE_VIEWER_DEFAULT_TAB: Default tab ("trace" or "json")
    
    Returns:
        DashboardConfig instance
    """
    trace_config = TraceViewerConfig(
        enabled=_parse_bool(os.getenv("TRACE_VIEWER_ENABLED", "1"), True),
        max_json_size_warning=int(os.getenv("TRACE_VIEWER_MAX_JSON_SIZE", "1000000")),
        show_token_usage=_parse_bool(os.getenv("TRACE_VIEWER_SHOW_TOKENS", "1"), True),
        default_tab=os.getenv("TRACE_VIEWER_DEFAULT_TAB", "trace"),
    )
    
    return DashboardConfig(
        experiments_db_path=os.getenv("EXPERIMENTS_DB_PATH"),
        onedrive_db_url=os.getenv(
            "ONEDRIVE_DB_URL",
            "https://grupoarpada-my.sharepoint.com/:u:/p/pcuervo/IQAl8U_XCr2iSJSTQE4kHYwIAU_go9Hkiktksk4RsO2veXs?e=VICW8o",
        ),
        always_refresh_db=_parse_bool(os.getenv("ALWAYS_REFRESH_DB", "0"), False),
        trace_viewer=trace_config,
    )


# Global config instance (loaded once at import)
_config: Optional[DashboardConfig] = None


def get_config() -> DashboardConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def is_trace_viewer_enabled() -> bool:
    """Check if the trace viewer is enabled."""
    return get_config().trace_viewer.enabled
