import logging
import threading

import colorlog

colorlog.basicConfig(encoding='utf-8', level=logging.INFO)

_logs = {}
_lock = threading.Lock()
_handlers = []


def set_levels(levels: dict[str, int]) -> None:
    for logger, level in levels.items():
        logging.getLogger(logger).setLevel(level)


# Minimize kafka log chattiness.
set_levels({'kafka': logging.WARNING})


def add_default_handler(h: logging.Handler):
    _handlers.append(h)

    for logger in _logs.values():
        logger.addHandler(h)


def get_logger(app_name: str | None = None) -> logging.Logger:
    n = f'app.{app_name}'
    d = len(n)
    log_name = f'{n:>20}' if d < 20 else f'~{n[-19:]}'

    with _lock:
        if log_name in _logs:
            return _logs[log_name]

    log_handler = colorlog.StreamHandler()
    log_handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(levelname).3s:%(name)s:%(asctime)s: %(message)s',
            datefmt='%m-%d-%Y %H:%M:%S',
            log_colors={
                'DEBUG': 'blue',
                'INFO': 'cyan',
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




