from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from logging_config import configure_logging


class LoggingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root_logger = logging.getLogger()
        self.original_handlers = self.root_logger.handlers[:]
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
            handler.close()

    def tearDown(self) -> None:
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            handler.close()
        for handler in self.original_handlers:
            logging.getLogger().addHandler(handler)

    def test_configure_logging_writes_utf8_file_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory) / "logs" / "app.log"
            configure_logging(log_file)
            configure_logging(log_file)
            logging.getLogger("test.system").info("系统启动")
            for handler in logging.getLogger().handlers:
                handler.flush()

            self.assertTrue(log_file.exists())
            self.assertEqual(log_file.read_text(encoding="utf-8").count("系统启动"), 1)
            file_handlers = [
                handler
                for handler in logging.getLogger().handlers
                if getattr(handler, "_pp_human_marker", False)
            ]
            self.assertEqual(len(file_handlers), 1)
            self._close_configured_handlers()

    def test_configure_logging_rotates_large_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory) / "app.log"
            configure_logging(log_file)
            logger = logging.getLogger("test.rotation")
            message = "x" * 1000
            for _ in range(6000):
                logger.info(message)
            for handler in logging.getLogger().handlers:
                handler.flush()

            self.assertTrue(log_file.exists())
            self.assertTrue(Path(f"{log_file}.1").exists())
            self.assertLessEqual(len(list(log_file.parent.glob("app.log.*"))), 3)
            self._close_configured_handlers()

    def _close_configured_handlers(self) -> None:
        for handler in logging.getLogger().handlers[:]:
            if getattr(handler, "_pp_human_marker", False):
                logging.getLogger().removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    unittest.main()
