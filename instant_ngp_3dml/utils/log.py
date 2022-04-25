# /usr/bin/python3
"""Log Tools."""
import logging
import os
import sys

logger = logging.getLogger('3dml-instant-ngp')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(message)s')

h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.DEBUG)
h1.setFormatter(formatter)
logger.propagate = False
logger.addHandler(h1)

def add_log_file(path: str, level: int = logging.DEBUG, append: bool = False):
    """Add Log output in file."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    _fh = logging.FileHandler(path, 'a' if append else 'w')
    _fh.setLevel(level)
    _fh.setFormatter(formatter)
    logger.addHandler(_fh)
