import argparse
import os
from src.logUtils import LogEntry


def get_files(base):
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith(".qdimacs"):
                yield os.path.join(root, file)


def has_result(file: str | os.PathLike) -> bool:
    log_dir = "log"
    basname = os.path.basename(file).split(".")[0:-1]
    basname = ".".join(basname)

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".pkl"):
                filename = os.path.join(root, file)
                file = file.split(".")[0:-1]
                file = ".".join(file)
                if basname == file:
                    try:
                        log_obj = LogEntry.from_file(filename)
                    except Exception as e:
                        print(f"Found bad log: {filename}, Error: {e}")
                        return False

                    if log_obj.exit_at_progress:
                        print(f"Found exit_at_progress: {filename}")
                        return False

                    if log_obj.exit_after_error:
                        print(f"Found exit_after_error: {filename}")
                        return False

                    if not any([log_obj.exit_after_preprocess,
                                log_obj.exit_after_leanskf,
                                log_obj.exit_after_refine,
                                log_obj.exit_after_timeout,
                                log_obj.exit_after_expection
                                ]):
                        print(f"Found unfinished: {filename}")
                        return False

                    return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "task_list", type=str, default="list.txt", help="list for task"
    )
    parser.add_argument(
        "output", type=str, default="tasks.list", help="output file name"
    )

    format_str = "python manthan.py --adaptivesample 1 --multiclass --lexmaxsat --unique 1 "

    args = parser.parse_args()
    fd = open(args.output, "w")
    total = 0
    todo = 0
    input = open(args.task_list, "r")
    lines = input.readlines()

    for file in lines:
        total += 1
        file = file.strip("\n")
        file = "dataset/" + file + ".qdimacs"
        if has_result(file):
            continue

        todo += 1
        cmd = format_str + file
        fd.write(cmd + "\n")

    fd.close()
    input.close()
    print(f"Total tasks: {total}, todo: {todo}")