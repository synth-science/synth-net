import sys
import time
import logging
import signal
import functools
from typing import Callable

def setup_logger(name, log_level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('logfile.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class TimeoutError(Exception):
    pass

def timeout_method(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        def handler(signum, frame):
            raise TimeoutError(f"Method '{method.__name__}' timed out after {self.timeout_seconds} seconds")

        original_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout_seconds)

        try:
            return method(self, *args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

    return wrapper

def timeit(method: Callable) -> Callable:

    @functools.wraps(method)
    def timed(*args, **kwargs):

        self = args[0]

        if not hasattr(self, '_runtimes'):
            self._runtimes = {}

        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()

        method_name = method.__name__
        runtime = end - start

        if method_name not in self._runtimes:
            self._runtimes[method_name] = []

        self._runtimes[method_name].append(runtime)

        return result

    return timed