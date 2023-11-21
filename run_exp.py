import argparse
import configparser
import copy
import multiprocessing as mp
import os

from manthan import manthan
from src.cirno.cirno import CirnoPool


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_files(base):
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith(".qdimacs"):
                yield os.path.join(root, file)


def has_result(file: str | os.PathLike) -> bool:
    out_dir = "out"
    log_dir = "log"
    basname = os.path.basename(file).split(".")[0:-1]
    basname = ".".join(basname)
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.endswith("_skolem.v"):
                if basname in file:
                    return True

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".pkl"):
                file = file.split(".")[0:-1]
                file = ".".join(file)
                if basname == file:
                    return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        help="it fixes the seed value, default is 10",
        default=10,
        dest="seed",
    )
    parser.add_argument(
        "--verb",
        type=int,
        help=" higher verb ensures higher verbose = 0 ,1 ,2, default is 1",
        default=1,
        dest="verbose",
    )
    parser.add_argument(
        "--gini",
        type=float,
        help="minimum impurity drop to prune the decision trees, default = 0.005",
        default=0.005,
        dest="gini",
    )

    parser.add_argument(
        "--maxrepairitr",
        type=int,
        default=5000,
        help="maximum allowed repair iterations; default 5000",
        dest="maxrepairitr",
    )

    parser.add_argument(
        "--adaptivesample",
        type=int,
        default=1,
        help="required to enable = 1/disable = 0  adaptive weighted sampling. Default is 1 ",
        dest="adaptivesample",
    )
    parser.add_argument(
        "--showtrees",
        action="store_true",
        help="To see generated decision trees as images use --showtrees",
    )
    parser.add_argument(
        "--maxsamples",
        type=int,
        help="num of samples used to learn the candidates. Takes int value. If not used, Manthan will decide value as per |Y|",
        dest="maxsamples",
    )
    parser.add_argument(
        "--preprocess",
        type=int,
        help=" to enable (=1) or disable (=0) unate function finding. Default 1",
        default=1,
        dest="preprocess",
    )
    parser.add_argument(
        "--multiclass",
        help="to learn a subset of existentially quantified variables together use --multiclass ",
        action="store_true",
    )
    parser.add_argument(
        "--lexmaxsat",
        help="to use lexicographical maxsat to find candidates to repair use --lexmaxsat ",
        action="store_true",
    )
    parser.add_argument(
        "--henkin",
        help="if you have dqdimacs instead of qdimacs, and would like to learn Henkin functions use --henkin",
        action="store_true",
    )
    parser.add_argument(
        "--logtime",
        help="to log time taken in each phase of manthan in <inputfile>_timedetails file use --logtime",
        action="store_true",
    )
    parser.add_argument(
        "--hop",
        help="if learning candidates via multiclassification, hop distances in primal graph is used to cluster existentially quantified variables together, use --hop <int> to define hop distance. Default is 3",
        type=int,
        default=3,
        dest="hop",
    )
    parser.add_argument(
        "--clustersize",
        type=int,
        help="maximum number of existentially quantified variables in a subset to be learned together via multiclassfication. Default is 8",
        default=8,
        dest="clustersize",
    )
    parser.add_argument(
        "--unique",
        help=" to enable (=1) or disable (=0) unique function finding. Default 1",
        type=int,
        default=1,
        dest="unique",
    )
    parser.add_argument("--workers", type=int, default=0, dest="workers")
    parser.add_argument("--timeout", type=int, default=7200, dest="timeout")
    parser.add_argument("input", help="input dir")

    args = parser.parse_args()
    if args.workers == 0:
        args.workers = int(mp.cpu_count() / 2)

    config = configparser.ConfigParser()
    configFilePath = "manthan_dependencies.cfg"
    config.read(configFilePath)

    mkdir("out")
    process_pool = CirnoPool(max_process=args.workers, is_smart=False)
    result_list = {}

    task_count = 0
    total_count = 0
    for file in get_files(args.input):
        total_count += 1
        if has_result(file):
            print(f"Instane: {file}'s result is exist")
            continue
        task_count += 1
        manthan_args = copy.deepcopy(args)
        manthan_args.input = file
        f = process_pool.submit(manthan, manthan_args, config)
        result_list[str(file)] = f
    print(f"Total: {total_count}, task: {task_count}")
    process_pool.shutdown()
    process_pool.close()

    for input, f in result_list.items():
        try:
            f.result
        except TimeoutError:
            print("YY" + input + "Timeout ERROR")
        except Exception as e:
            print("YY" + input + " " + e)
