# /usr/bin/python3
"""Json Loader."""
import json
import os
from enum import Enum

import numpy as np


class NpEncoder(json.JSONEncoder):
    """Encode numpy in json."""
    # pylint: disable=super-with-arguments

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)

        if isinstance(o, np.floating):
            return float(o)

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, tuple):
            return str(o)

        if isinstance(o, Enum):
            return o.value

        return super(NpEncoder, self).default(o)


def write_json(path, dictionnary_content, pretty: bool = False):
    """Save json file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as _f:
        if pretty:
            json.dump(dictionnary_content, _f, indent=4, cls=NpEncoder)
        else:
            json.dump(dictionnary_content, _f, cls=NpEncoder)


def read_json(path) -> dict:
    """Save json file."""
    with open(path, "r", encoding="utf-8") as _f:
        return json.load(_f)
