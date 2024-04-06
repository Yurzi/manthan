import argparse
import os
from typing import Generator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.logUtils import LogEntry

LOGFILE_DIR = "log"
LOGFILE_ENDWITH = ".pkl.zst"
OUTFILE_DIR = "out"
OUTFILE_ENDWITH = "_skolem.v"

progress_bar = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
)


def get_files(base: os.PathLike | str) -> Generator[os.PathLike | str, None, None]:
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith(".qdimacs"):
                yield os.path.join(root, file)


def get_instance_name(qdimacs_path: os.PathLike | str) -> str:
    path = os.path.basename(qdimacs_path)
    instance_name = path.split(".")[0:-1]
    instance_name = ".".join(instance_name)
    return instance_name


def has_outfile(instance_name: str) -> bool:
    outfile = os.path.join(OUTFILE_DIR, instance_name + OUTFILE_ENDWITH)
    return os.path.exists(outfile)


def has_logfile(instance_name: str) -> bool:
    logfile = os.path.join(LOGFILE_DIR, instance_name + LOGFILE_ENDWITH)
    return os.path.exists(logfile)


def has_error(instance_name: str) -> bool:
    logfile = os.path.join(LOGFILE_DIR, instance_name + LOGFILE_ENDWITH)
    try:
        logEntry: LogEntry = LogEntry.from_file(logfile)
    except Exception:
        progress_bar.console.print("    [red]Error[/red]: bad logfile")
        return True

    if logEntry.exit_after_error:
        progress_bar.console.print("    [red]Error[/red]: exit after error")
        return True

    if logEntry.exit_after_expection:
        progress_bar.console.print("    [red]Error[/red]: exit after expection")
        return True

    if logEntry.exit_at_progress:
        progress_bar.console.print("    [red]Error[/red]: exit at progress")
        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "task_dir", type=str, default="dataset", help="directory to store qdimacs files"
    )
    parser.add_argument(
        "output", type=str, default="tasks.list", help="output file name"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="force to generate tasks list",
        dest="force",
    )

    parser.add_argument(
        "--skip-no-out",
        action="store_true",
        default=False,
        help="skip task without output file",
        dest="skip_no_out",
    )

    parser.add_argument(
        "--skip-has-error",
        action="store_true",
        default=False,
        help="skip task with error in log file",
        dest="skip_has_error",
    )

    parser.add_argument(
        "--ignore-out",
        action="store_true",
        default=False,
        help="ignore out file",
        dest="ignore_out",
    )

    format_str = (
        "python manthan.py --adaptivesample 1 --multiclass --lexmaxsat --unique 1 "
    )

    args = parser.parse_args()

    files = [file for file in get_files(args.task_dir)]
    total = len(files)
    files_tqdm = progress_bar.add_task("[blue] Checking...", total=total)
    progress_bar.start()

    fd = open(args.output, "w")
    todo = 0
    for file in files:
        instance_name = get_instance_name(file)
        progress_bar.print(f"[green]Now[/green]: {instance_name}")
        if args.force:
            todo += 1
            cmd = format_str + str(file)
            fd.write(cmd + "\n")
            progress_bar.advance(files_tqdm)
            continue
        # else
        if not args.ignore_out:
            if has_outfile(instance_name):
                progress_bar.console.print("    [green]Skip[/green]: has out")
                progress_bar.advance(files_tqdm)
                continue

        if has_logfile(instance_name):
            if args.skip_has_error or not has_error(instance_name):
                progress_bar.console.print("    [green]Skip[/green]: no error")
                progress_bar.advance(files_tqdm)
                continue

        todo += 1
        cmd = format_str + str(file)
        fd.write(cmd + "\n")
        progress_bar.advance(files_tqdm)

    fd.close()
    progress_bar.stop()
    print(f"Total tasks: {total}, todo: {todo}")
