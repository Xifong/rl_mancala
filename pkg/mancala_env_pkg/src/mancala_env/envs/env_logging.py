import logging

env_logger = logging.getLogger(__name__)
# setting warning level by default, in accordance with the advice on library logging
# see https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library (2025-03-02)
env_logger.setLevel(logging.WARNING)


def get_game_information_message_format() -> str:
    """
    Returns: A str representing how game specific information should be formatted in logs
    """
    return "Game ID:%(game_id)s"


def get_logger_with_context(extra: dict) -> logging.Logger:
    return logging.LoggerAdapter(env_logger, extra)
