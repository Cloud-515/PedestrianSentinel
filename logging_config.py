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
            # 带 BOM 的 UTF-8：日志是给人在 Windows 上直接打开看的，而 Windows
            # PowerShell 5.1 的 Get-Content 默认按 ANSI 解码，无 BOM 的 UTF-8 中文
            # 会整片变成乱码（排查现场问题时很容易被当成程序在乱写日志）。
            # 其它读日志的工具和编码探测都能正常处理 BOM。
            encoding="utf-8-sig",
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
