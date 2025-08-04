import logging
import threading

import colorlog

colorlog.basicConfig(encoding='utf-8', level=logging.INFO)

_logs = {}
_lock = threading.Lock()
_handlers = []


def create_levels(levels: dict[str, int]) -> None:
    for logger, level in levels.items():
        logging.getLogger(logger).setLevel(level)


def get_logger(app_name: str | None = None) -> logging.Logger:
    n = f'app.{app_name}'
    d = len(n)
    log_name = f'{n:>10}' if d < 10 else f'~{n[-9:]}'

    with _lock:
        if log_name in _logs:
            return _logs[log_name]

    log_handler = colorlog.StreamHandler()
    log_handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(levelname).3s:%(name)s:%(asctime)s: %(message)s',
            datefmt='%m-%d-%Y %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'blue',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )

    log = colorlog.getLogger(log_name)

    log.addHandler(log_handler)

    for h in _handlers:
        log.addHandler(h)

    log.propagate = False
    log.setLevel(logging.DEBUG)

    with _lock:
        if log_name in _logs:
            return _logs[log_name]
        _logs[log_name] = log

    return log




