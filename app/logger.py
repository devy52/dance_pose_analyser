# app/logger.py
import logging
from pathlib import Path

LOG_DIR = Path("app/logs")
LOG_FILE = LOG_DIR / "processor.log"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# If log file exists and is not empty, clear it
'''if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
    LOG_FILE.write_text("")'''

# If log file exists
if LOG_FILE.exists():
    LOG_FILE.write_text("")

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler()  # also print to console
    ]
)

# Create a named logger (optional but cleaner)
logger = logging.getLogger("app_logger")
logger.info("=== New Logging Session Started ===")
