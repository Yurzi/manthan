import os
import sys
import argparse
import configparser
import copy
import multiprocessing as mp

from manthan import manthan

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_files(base):
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith(".qdimacs"):
                yield os.path.join(root, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed', type=int, help="it fixes the seed value, default is 10", default=10, dest='seed')
    parser.add_argument(
        '--verb', type=int, help=" higher verb ensures higher verbose = 0 ,1 ,2, default is 1", default=1, dest='verbose')
    parser.add_argument(
        '--gini', type=float, help="minimum impurity drop to prune the decision trees, default = 0.005", default=0.005, dest='gini')

    parser.add_argument('--maxrepairitr', type=int, default=5000,
                        help="maximum allowed repair iterations; default 5000", dest='maxrepairitr')

    parser.add_argument('--adaptivesample', type=int, default=1,
                        help="required to enable = 1/disable = 0  adaptive weighted sampling. Default is 1 ", dest='adaptivesample')
    parser.add_argument('--showtrees', action='store_true',
                        help="To see generated decision trees as images use --showtrees")
    parser.add_argument('--maxsamples', type=int,
                        help="num of samples used to learn the candidates. Takes int value. If not used, Manthan will decide value as per |Y|", dest='maxsamples')
    parser.add_argument("--preprocess", type=int,
                        help=" to enable (=1) or disable (=0) unate function finding. Default 1", default=1, dest='preprocess')
    parser.add_argument(
        "--multiclass", 
        help="to learn a subset of existentially quantified variables together use --multiclass ", action='store_true')
    parser.add_argument(
        "--lexmaxsat", help="to use lexicographical maxsat to find candidates to repair use --lexmaxsat ", action='store_true')
    parser.add_argument(
        "--henkin", help="if you have dqdimacs instead of qdimacs, and would like to learn Henkin functions use --henkin", action='store_true')
    parser.add_argument(
        "--logtime", help="to log time taken in each phase of manthan in <inputfile>_timedetails file use --logtime", action='store_true')
    parser.add_argument("--hop",  help="if learning candidates via multiclassification, hop distances in primal graph is used to cluster existentially quantified variables together, use --hop <int> to define hop distance. Default is 3", type=int, default=3, dest='hop')
    parser.add_argument("--clustersize", type=int,
                        help="maximum number of existentially quantified variables in a subset to be learned together via multiclassfication. Default is 8", default=8, dest='clustersize')
    parser.add_argument("--unique", help=" to enable (=1) or disable (=0) unique function finding. Default 1",
                        type=int, default=1, dest='unique')
    parser.add_argument("--workers", type=int, default=0, dest='workers')

    parser.add_argument("input", help="input dir")

    args = parser.parse_args()
    if args.workers == 0:
        args.workers = mp.cpu_count()

    config = configparser.ConfigParser()
    configFilePath = "manthan_dependencies.cfg"
    config.read(configFilePath)
        
    mkdir("out")
    

    process_list = []
    for file in get_files(args.input):
        manthan_args = copy.deepcopy(args)
        manthan_args.input = file
        process = mp.Process(target=manthan, args=(manthan_args, config, ))
        process_list.append(process)

    workers = args.workers

    tasks = len(process_list)
    idle_workers = workers
    working = []
    pendding_id = 0

    while True:
        if tasks == 0:
            break
        if idle_workers > 0:
            for i in range(idle_workers):
                if pendding_id >= len(process_list):
                    break
                process = process_list[pendding_id]
                process.start()
                working.append(process)
                idle_workers -= 1
                pendding_id += 1


        for process in working:
            if not process.is_alive():
                process.join()
                working.remove(process)
                idle_workers += 1
                tasks -= 1
                process.close()