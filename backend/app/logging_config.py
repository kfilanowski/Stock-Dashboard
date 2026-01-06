"""
Logging configuration for the Stock Dashboard API.

Provides structured logging with different levels for development and production.
"""
import logging
import sys
from typing import Optional
from .config import settings


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.
    
    Args:
        level: Override log level. Defaults to DEBUG if settings.debug, else INFO.
    """
    if level is None:
        level = "DEBUG" if settings.debug else "INFO"
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name, typically __name__ from the calling module.
        
    Returns:
        Configured logger instance.
        
    Example:
        from .logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Processing request")
    """
    return logging.getLogger(name)

