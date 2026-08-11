from __future__ import annotations

import logging
import sys


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    configure_logging()
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "未安装 PySide6。请运行: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        from main_window import MainWindow
    except ModuleNotFoundError as error:
        print(
            f"缺少运行依赖 {error.name!r}。请运行: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    application = QApplication(sys.argv)
    application.setApplicationName("行人警戒区域监控")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
