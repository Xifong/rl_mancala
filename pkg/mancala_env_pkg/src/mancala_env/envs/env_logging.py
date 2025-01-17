from typing import Tuple, Callable
import logging


def extract_root_logging_config() -> Tuple[str, str, int]:
    root_logger = logging.getLogger()

    assert root_logger.handlers, "could not find handlers on the root_logger"
    handler_count = len(root_logger.handlers)
    assert (
        handler_count == 1
    ), f"should only be 1 root_logger handler, instead found {handler_count}"
    handler = root_logger.handlers[0]
    assert (
        type(handler) == logging.FileHandler
    ), f"Expected logger to be of type FileHandler but was {type(handler)}"

    return handler.baseFilename, handler.formatter._fmt, handler.level


def get_game_id_formatter(format: str) -> logging.Formatter:
    split = format.split("-")
    assert (
        split[-1] == " %(message)s"
    ), f"format set in logging.basicConfig does not match expectation. Expected ending in '- %(message)s', got ending in '{split[-1]}'"
    new_format = "-".join(split[:-1]) + " - Game ID:%(game_id)s - %(message)s"

    return logging.Formatter(new_format)


def setup_env_logger() -> Callable[[dict], logging.Logger]:
    log_file, format, level = extract_root_logging_config()

    new_logger = logging.getLogger(__name__)
    new_logger.setLevel(level)
    new_logger.propagate = False
    if not new_logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = get_game_id_formatter(format)
        handler.setFormatter(formatter)
        new_logger.addHandler(handler)

    def get_logger_with_context(extra: dict) -> logging.Logger:
        return logging.LoggerAdapter(new_logger, extra)

    return get_logger_with_context
