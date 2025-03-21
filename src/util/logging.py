import logging

from rich.logging import RichHandler
from rich.traceback import install

install(show_locals=True, suppress=["torch", "torchvision"])

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")
