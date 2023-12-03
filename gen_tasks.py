import argparse
import os


def get_files(base):
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith(".qdimacs"):
                yield os.path.join(root, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "task_dir", type=str, default="dataset", help="directory to store qdimacs files"
    )
    parser.add_argument(
        "output", type=str, default="tasks.list", help="output file name"
    )

    format_str = "python manthan.py --adaptivesample 1 --multiclass --lexmaxsat --unique 1 "

    args = parser.parse_args()
    fd = open(args.output, "w")
    for file in get_files(args.task_dir):
        cmd = format_str + file
        fd.write(cmd + "\n")

    fd.close()
