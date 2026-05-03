import sys
from loguru import logger
from pathlib import Path

# Creating folder for logs
Path("logs").mkdir(parents=True, exist_ok=True)

# Setting rotation: file up to 5 MB, store 3 last backups
logger.add("logs/app.log", rotation="5 MB", retention=3, level="INFO")

# Exporting logger under the name log | avoid breaking old imports
logger = logger