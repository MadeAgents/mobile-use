import os
import logging
import colorama
import time
from typing import Any, Dict, Callable, List, Union
from colorama import Fore, Style
from logging.handlers import RotatingFileHandler

colorama.init()


class ColoredFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: Style.DIM + Fore.WHITE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Style.BRIGHT + Fore.RED,
    }
    converter = time.localtime
    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, callbacks: Union[Callable[[Any], Any], List[Callable[[Any], Any]]]=None):
        self.callbacks = callbacks if isinstance(callbacks, list) or callbacks is None else [callbacks]
        super().__init__(fmt,datefmt,style,validate)

    def format(self, record):
        log_color = ColoredFormatter.COLOR_MAP.get(record.levelno)
        super().format(record)
        msg = super().format(record)

        if hasattr(record, "color"):
            log_color = getattr(record, "color")

        title, t_color = None, None
        if hasattr(record, "title"):
            title = getattr(record, "title")
        else:
            title = ''
        if hasattr(record, "t_color"):
            t_color = getattr(record, "t_color")
        else:
            t_color = Fore.CYAN if record.levelno == logging.INFO else log_color
        if hasattr(record, "sep"):
            sep = getattr(record, "sep")
        else:
            sep = ' '
        no_color_message = title + sep + msg
        no_color_message = no_color_message.strip()
        if title:
            title = t_color + title + Style.RESET_ALL
        message = title + sep + log_color + msg + Style.RESET_ALL
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(no_color_message)
        if hasattr(record, "callback"):
            callback = getattr(record, "callback")
            callback_kwargs = {}
            if hasattr(record, "callback_kwargs"):
                callback_kwargs = getattr(record, "callback_kwargs")
            callback(no_color_message, **callback_kwargs)

        return message.lstrip()


def setup_logger(name: str=None, level: str='INFO', file_handler: Dict=None, callbacks: Union[Callable[[Any], Any], List[Callable[[Any], Any]]]=None):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    if level == 'OFF' or level is False:
        return
    level = level.upper()
    logger.setLevel(level=level)
    log_format = logging.Formatter('%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
    color_log = ColoredFormatter('%(message)s', callbacks=callbacks)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_log)
    console_handler.setLevel(level=level)

    if file_handler is not None:
        log_dir = file_handler.get('log_dir')
        os.makedirs(log_dir, exist_ok=True)
        file_name = file_handler.get('file_name', 'toolagent.log')
        log_file = os.path.join(log_dir, file_name)
        rotating_file_handler = RotatingFileHandler(log_file, maxBytes=file_handler.get('maxBytes', 1024*1024*5), backupCount=file_handler.get('backupCount', 5))
        rotating_file_handler.setFormatter(log_format)
        rotating_file_handler.setLevel(level=file_handler.get('level', level))
        logger.addHandler(rotating_file_handler)
    logger.addHandler(console_handler)
