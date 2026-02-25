import logging
from logging import handlers
from logging.handlers import RotatingFileHandler
import structlog
import sys
from pathlib import Path

def setup_logger():
    # Creating logs folder (if not exists)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Rotation
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=5_000_000, backupCount=3
    )
    file_handler.setLevel(logging.INFO)

    # Console settings
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Logger settings
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True
    )
    
    # STRUCTLOG settings
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()

log = setup_logger()