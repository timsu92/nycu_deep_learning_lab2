import logging

from rich.logging import RichHandler
from rich.traceback import install
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

install(show_locals=True, suppress=["torch", "torchvision"])

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")

job_progress = lambda: Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(
        style="bar.back",
        complete_style="bar.complete",
        finished_style="bar.finished",
        pulse_style="bar.pulse",
    ),
    TaskProgressColumn(show_speed=True),
    TextColumn("ETA"),
    TimeRemainingColumn(elapsed_when_finished=True),
    TextColumn("Elapsed"),
    TimeElapsedColumn(),
)
