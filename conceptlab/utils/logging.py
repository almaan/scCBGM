import logging


def setup_logger(log_level=logging.INFO, log_file=None):
    # Create or retrieve the custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)  # Set the log level for the logger

    # Child Logger 1: Has its own handler
    child_logger1 = logging.getLogger(
        "conceptlab"
    )  # This logger is a child of the root logger
    child_handler1 = logging.StreamHandler()
    child_handler1.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    child_handler1.setFormatter(formatter)

    child_logger1.addHandler(child_handler1)
    child_logger1.setLevel(logging.DEBUG)

    child_logger1.propagate = False

    return child_logger1
