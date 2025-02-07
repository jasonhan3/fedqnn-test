import logging
def get_logger(log_level, name="my_logger"):
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # Set the minimum level for this logger

    return logger