from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import app_paths

APP_DIR = app_paths.APP_DIR
LOG_DIR = app_paths.data("logs")
LOG_FILE = LOG_DIR / "app.log"
_LOG_HANDLER_MARKER = "pp_human_file_handler"


def configure_logging(log_file: Path = LOG_FILE) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    has_file_handler = any(
        getattr(handler, "_pp_human_marker", False)
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        setattr(file_handler, "_pp_human_marker", _LOG_HANDLER_MARKER)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    for handler in root_logger.handlers:
        if not isinstance(handler, RotatingFileHandler) or not getattr(
            handler, "_pp_human_marker", False
        ):
            if handler.formatter is None:
                handler.setFormatter(formatter)

    logging.captureWarnings(True)
