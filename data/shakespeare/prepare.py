import os

import requests
from loguru import logger

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = "input.txt"


if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, "w") as f:
        f.write(requests.get(DATA_URL).text)

with open(DATA_PATH, "r") as f:
    data = f.read()

logger.info(f"Length of dataset in characters: {len(data)}")

logger.info(f"Sample text:\n{data[:1000]}")
