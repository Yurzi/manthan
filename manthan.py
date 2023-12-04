#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Priyanka Golia, Subhajit Roy, and Kuldeep Meel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import print_function

import argparse
import atexit
import configparser
import os
import sys
import tempfile
import time
import signal
import psutil

import networkx as nx
import numpy as np

from src.candidateSkolem import learnCandidate
from src.convertVerilog import convert_verilog
from src.createSkolem import (addSkolem, createErrorFormula, createSkolem,
                              createSkolemfunction, skolemfunction_preprocess,
                              verify)
from src.generateSamples import computeBias, generatesample
from src.logUtils import (LogEntry, get_from_file, get_inputfile_contenet,
                          set_run_pid, unset_run_pid)
from src.preprocess import convertcnf, parse, preprocess
from src.repair import (addXvaluation, callMaxsat, callRC2, maxsatContent,
                        repair, updateSkolem)


log_entry = LogEntry()


def handle_exit(signum, frame):
    print(" c Manthan interrupted")
    log_entry.exit_at_progress = True
    p = psutil.Process()
    for child in p.children(recursive=True):
        child.terminate()

    sys.exit(-2)


def handle_timout(signum, frame):
    print(" c Manthan timed out")
    log_entry.exit_after_timeout = True
    p = psutil.Process()
    for child in p.children(recursive=True):
        child.terminate()

    sys.exit(-1)


def logtime(inputfile, text):
    with open(inputfile + "time_details", "a+") as f:
        f.write(text + "\n")
    f.close()


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def check_config(config):
    if config.has_option("ITP-Path", "itp_path"):
        sys.path.append(config["ITP-Path"]["itp_path"])
        # from src.callUnique import find_unique_function
    else:
        print("c could not find itp module")
        print("c check unique installation")
        print(
            "c check if pythonblind[global] is installed and found in cmake of unique"
        )
        exit()

    if not config.has_section("Dependencies-Path"):
        print(
            " c Did not install dependencies from source code,",
            "using precomplied binaries"
        )
        config.add_section("Dependencies-Path")
        config.set(
            "Dependencies-Path", "openwbo_path", "./dependencies/static_bin/open-wbo"
        )
        config.set(
            "Dependencies-Path", "cmsgen_path", "./dependencies/static_bin/cmsgen"
        )
        config.set(
            "Dependencies-Path", "picosat_path", "./dependencies/static_bin/picosat"
        )
        config.set(
            "Dependencies-Path",
            "preprocess_path",
            "./dependencies/static_bin/preprocess",
        )
        config.set(
            "Dependencies-Path",
            "file_generation_cex_path",
            "./dependencies/static_bin/file_generation_cex",
        )

    else:
        if not config.has_option("Dependencies-Path", "openwbo_path"):
            config.set(
                "Dependencies-Path",
                "openwbo_path",
                "./dependencies/static_bin/open-wbo",
            )

        if not config.has_option("Dependencies-Path", "cmsgen_path"):
            config.set(
                "Dependencies-Path", "cmsgen_path", "./dependencies/static_bin/cmsgen"
            )

        if not config.has_option("Dependencies-Path", "picosat_path"):
            config.set(
                "Dependencies-Path", "picosat_path", "./dependencies/static_bin/picosat"
            )

        if not config.has_option("Dependencies-Path", "preprocess_path"):
            config.set(
                "Dependencies-Path",
                "preprocess_path",
                "./dependencies/static_bin/preprocess",
            )

        if not config.has_option("Dependencies-Path", "file_generation_cex_path"):
            config.set(
                "Dependencies-Path",
                "file_generation_cex_path",
                "./dependencies/static_bin/file_generation_cex",
            )

    return config


def manthan(args, config, queue=None):
    config = check_config(config)
    if config.has_option("ITP-Path", "itp_path"):
        sys.path.append(config["ITP-Path"]["itp_path"])
        from src.callUnique import find_unique_function

    print(" c parsing")
    start_time = time.time()

    if args.henkin:
        Xvar, Yvar, HenkinDep, qdimacs_list, dg = parse(args)

    else:
        Xvar, Yvar, qdimacs_list, dg = parse(args)

    # log
    log_entry.input_vars = Xvar
    log_entry.output_vars = Yvar
    log_entry.dag = dg
    log_entry.clause_list = qdimacs_list
    if args.henkin:
        log_entry.henkin_dep = HenkinDep

    """
        We create a DAG to handle dependencies among existentially quantified variables
        if y_i depends on y_j, there is a edge from y_i to y_j
    """

    if args.verbose:
        print(" c count X (universally quantified variables) variables", len(Xvar))
        print(" c count Y (existentially quantified variables) variables", len(Yvar))

    if args.verbose >= 2:
        print(" c  X (universally quantified variables) variables", (Xvar))
        print(" c  Y (existentially quantified variables) variables", (Yvar))

    inputfile_name = args.input.split("/")[-1].split(".")[:-1]
    inputfile_name = ".".join(inputfile_name)
    print(" c inputfile_name", inputfile_name)
    log_entry.instance_name = inputfile_name
    log_entry.instance_str = get_inputfile_contenet(args.input)

    cnffile_name = tempfile.gettempdir() + "/" + inputfile_name + ".cnf"

    if not args.henkin:
        cnfcontent = convertcnf(args, cnffile_name)
    else:
        cnfcontent = convertcnf(args, cnffile_name, Yvar)

    cnfcontent = cnfcontent.strip("\n") + "\n"
    log_entry.cnfcontent = cnfcontent.strip("\n")

    if (args.preprocess) and (not (args.henkin)):
        print(" c preprocessing: finding unates (constant functions)")
        start_time_preprocess = time.time()

        """
        We find constants functions
        only if the existentially quantified variables are less then 20000
        else it takes way much time to find the constant functions.
        """

        if len(Yvar) < 20000:
            PosUnate, NegUnate = preprocess(cnffile_name, args, config)
        else:
            print(" c too many Y variables, let us proceed with Unique extraction\n")
            PosUnate = []
            NegUnate = []

        end_time_preprocess = time.time()

        if args.logtime:
            logtime(
                inputfile_name,
                "preprocessing time:"
                + str(end_time_preprocess - start_time_preprocess),
            )

        if args.verbose:
            print(" c count of positive unates", len(PosUnate))
            print(" c count of negative unates", len(NegUnate))
            if args.verbose >= 2:
                print(" c positive unates", PosUnate)
                print(" c negative unates", NegUnate)

        Unates = PosUnate + NegUnate

        """
        Adding unates in the specification as constants.
        This might help samplers to produce good data

        """

        for yvar in PosUnate:
            qdimacs_list.append([yvar])
            cnfcontent += "%s 0\n" % (yvar)

        for yvar in NegUnate:
            qdimacs_list.append([-1 * int(yvar)])
            cnfcontent += "-%s 0\n" % (yvar)

        log_entry.perprocess_out = Unates
        log_entry.preprocess_time = end_time_preprocess - start_time_preprocess

    else:
        Unates = []
        PosUnate = []
        NegUnate = []
        print(
            " c preprocessing is disabled. To do preprocessing, please use --preprocess"
        )

    if len(Unates) == len(Yvar):
        print(" c all Y variables are unates and have constant functions")

        """
        Generating verilog files to output Skolem functions.
        """

        log_entry.output_verilog = skolemfunction_preprocess(
            inputfile_name, Xvar, Yvar, PosUnate, NegUnate
        )

        end_time = time.time()
        log_entry.total_time = end_time - start_time
        log_entry.exit_after_preprocess = True

        print(" c Manthan has synthesized Skolem functions")
        print(" c Total time taken", str(end_time - start_time))
        print("Skolem functions are stored at %s_skolem.v" % (inputfile_name))

        if args.logtime:
            logtime(inputfile_name, "totaltime:" + str(end_time - start_time))

        return
        # exit()

    if args.unique:
        print(" c finding uniquely defined functions")

        start_time_unique = time.time()

        if args.henkin:
            UniqueVars, UniqueDef, dg = find_unique_function(
                args, qdimacs_list, Xvar, Yvar, dg, Unates, HenkinDep
            )
        else:
            UniqueVars, UniqueDef, dg = find_unique_function(
                args, qdimacs_list, Xvar, Yvar, dg, Unates
            )

        end_time_unique = time.time()

        if args.logtime:
            logtime(
                inputfile_name,
                "unique function finding:" + str(end_time_unique - start_time_unique),
            )

        if args.verbose:
            print(" c count of uniquely defined variables", len(UniqueVars))

            if args.verbose >= 2:
                print(" c uniquely defined variables", UniqueVars)
    else:
        UniqueVars = []
        UniqueDef = ""
        print(
            " c finding unique function is disabled.",
            "To find unique functions please use -- unique"
        )

    if len(Unates) + len(UniqueVars) == len(Yvar):
        print(" c all Y variables are either unate or unique")
        print(" c found functions for all Y variables")

        log_entry.output_verilog = skolemfunction_preprocess(
            inputfile_name, Xvar, Yvar, PosUnate, NegUnate, UniqueVars, UniqueDef
        )

        end_time = time.time()
        log_entry.total_time = end_time - start_time
        log_entry.exit_after_preprocess = True

        print(" c Total time taken", str(end_time - start_time))
        print("Skolem functions are stored at %s_skolem.v" % (inputfile_name))

        if args.logtime:
            logtime(inputfile_name, "totaltime:" + str(end_time - start_time))

        return

    """
    deciding the number of samples to be generated
    """
    start_time_datagen = time.time()

    if not args.maxsamples:
        if len(Xvar) > 4000:
            num_samples = 1000
        if (len(Xvar) > 1200) and (len(Xvar) <= 4000):
            num_samples = 5000
        if len(Xvar) <= 1200:
            num_samples = 10000
    else:
        num_samples = args.maxsamples

    log_entry.num_samples = num_samples

    """
    We can either choose uniform sampler or weighted sampler.
    In case of weighted sampling,
    we need to find adaptive weights for each positive literals
    including X and Y.
    """

    print(" c generating samples")

    sampling_cnf = cnfcontent

    if args.adaptivesample:
        """
        sampling_weights_y_1 is to bias outputs towards 1
        sampling_weights_y_o is to bias outputs towards 0

        """

        sampling_weights_y_1 = ""
        sampling_weights_y_0 = ""
        for xvar in Xvar:
            sampling_cnf += "w %s 0.5\n" % (xvar)
        for yvar in Yvar:
            """
            uniquely defined variables are now treated as "inputs"
            """

            if yvar in UniqueVars:
                sampling_cnf += "w %s 0.5\n" % (yvar)
                continue

            if (yvar in PosUnate) or (yvar in NegUnate):
                continue

            sampling_weights_y_1 += "w %s 0.9\n" % (yvar)
            sampling_weights_y_0 += "w %s 0.1\n" % (yvar)

        if args.verbose >= 2:
            print(" c computing adaptive bias for Y variables")
            print(" c sampling_weights_y_1 %s" % (sampling_weights_y_1))
            print(" c sampling_weights_y_0 %s" % (sampling_weights_y_0))
            print(" c sampling cnf", sampling_cnf)
            print(" c inputfile_name", inputfile_name)

        weighted_sampling_cnf = computeBias(
            args,
            config,
            Yvar,
            sampling_cnf,
            sampling_weights_y_1,
            sampling_weights_y_0,
            inputfile_name,
            Unates + UniqueVars,
        )

        if args.verbose >= 2:
            print(" c generating samples..")

        samples = generatesample(
            args, config, num_samples, weighted_sampling_cnf, inputfile_name
        )
    else:
        print(" c generating uniform samples")
        samples = generatesample(
            args, config, num_samples, sampling_cnf, inputfile_name
        )

    end_time_datagen = time.time()

    log_entry.datagen_out = samples
    log_entry.datagen_time = end_time_datagen - start_time_datagen

    if args.logtime:
        logtime(
            inputfile_name,
            "generating samples:" + str(end_time_datagen - start_time_datagen),
        )

    print(" c generated samples.. learning candidate functions via decision learning")

    """
    we need verilog file for repairing the candidates,
    hence first let us convert the qdimacs to verilog
    ng is used only if we are doing multiclassification.
    It has an edge only if y_i and y_j share a clause
    this is used to cluster the variables for which functions could be learned together.
    """

    verilogformula, dg, ng = convert_verilog(args, Xvar, Yvar, dg)

    start_time_learn = time.time()

    if not args.henkin:
        candidateSkf, dg = learnCandidate(
            Xvar, Yvar, UniqueVars, PosUnate, NegUnate, samples, dg, ng, args
        )
    else:
        candidateSkf, dg = learnCandidate(
            Xvar, Yvar, UniqueVars, PosUnate, NegUnate, samples, dg, ng, args, HenkinDep
        )

    end_time_learn = time.time()

    if args.logtime:
        logtime(
            inputfile_name,
            "candidate learning:" + str(end_time_learn - start_time_learn),
        )
    log_entry.leanskf_time = end_time_learn - start_time_learn

    """
    YvarOrder is a total order of Y variables that represents interdependecies among Y.
    """

    YvarOrder = np.array(list(nx.topological_sort(dg)))

    assert len(Yvar) == len(YvarOrder)

    """

    createSkolem here represents candidate Skolem functions for each Y variables
    in a verilog format.

    """
    createSkolem(candidateSkf, Xvar, Yvar, UniqueVars, UniqueDef, inputfile_name)
    log_entry.leanskf_out = get_from_file(
        tempfile.gettempdir() + "/" + inputfile_name + "_skolem.v"
    )

    if args.verbose >= 2:
        print("c learned candidate functions", candidateSkf)

    error_content = createErrorFormula(Xvar, Yvar, verilogformula)
    log_entry.errorformula_out = error_content

    """
    We use maxsat to identify the candidates to repair.
    We are converting specification as a hard constraint for maxsat.
    """
    maxsatWt, maxsatcnf, cnfcontent = maxsatContent(
        cnfcontent, (len(Xvar) + len(Yvar)), (len(PosUnate) + len(NegUnate))
    )

    countRefine = 0

    start_time_repair = time.time()

    while True:
        now_time = time.time()

        """
        adding Y' <-> f(X) term in the error formula
        """

        addSkolem(error_content, inputfile_name)

        """
        sigma [0]: valuation for X
        sigma [1]: valuation for Y
        sigma [2]: valuation for Y' where Y' <-> f(X)

        """

        check, sigma, ret = verify(args, config, Xvar, Yvar, inputfile_name)

        if check == 0:
            print(" c error --- ABC network read fail")
            log_entry.exit_after_error = True
            break

        if ret == 0:
            print(" c verification check UNSAT")

            if args.verbose:
                print(" c no more repair needed")
                print(" c number of repairs needed to converge", countRefine)

            log_entry.output_verilog = createSkolemfunction(inputfile_name, Xvar, Yvar)

            end_time = time.time()
            log_entry.sat = True

            print(" c Total time taken", str(end_time - start_time))
            print("Skolem functions are stored at %s_skolem.v" % (inputfile_name))
            break

        if ret == 1:
            if now_time - start_time > args.timeout:
                print(" c timeout")
                log_entry.exit_after_timeout = True
                break

            countRefine += 1  # update the number of repair itr

            if args.verbose > 1:
                print(" c verification check is SAT, we have counterexample to fix")
                print(" c number of repair", countRefine)
                print(" c finding candidates to repair using maxsat")

            if args.verbose >= 2:
                print("counter example to repair", sigma)

            repaircnf, maxsatcnfRepair = addXvaluation(
                cnfcontent, maxsatWt, maxsatcnf, sigma[0], Xvar
            )

            ind = callMaxsat(
                args,
                config,
                maxsatcnfRepair,
                sigma[2],
                UniqueVars + Unates,
                Yvar,
                YvarOrder,
                inputfile_name,
            )

            if args.verbose >= 2:
                print("candidates to repair", ind)

            assert len(ind) > 0

            if args.verbose > 1:
                print(" c number of candidates undergoing repair iterations", len(ind))

            if args.verbose >= 2:
                print(" c number of candidates undergoing repair iterations", len(ind))
                print(" c variables undergoing refinement", ind)

            if not args.henkin:
                lexflag, repairfunctions = repair(
                    args,
                    config,
                    repaircnf,
                    ind,
                    Xvar,
                    Yvar,
                    YvarOrder,
                    dg,
                    UniqueVars + Unates,
                    sigma,
                    inputfile_name,
                )
            else:
                lexflag, repairfunctions = repair(
                    args,
                    config,
                    repaircnf,
                    ind,
                    Xvar,
                    Yvar,
                    YvarOrder,
                    dg,
                    UniqueVars + Unates,
                    sigma,
                    inputfile_name,
                    HenkinDep,
                )

            """
            if we encounter too many repair candidates
            while repair the previous identifed candidates ind,

            we call lexmaxsat to identify nicer candidates to repair
            in accordance with the dependencies.
            """

            if lexflag and args.lexmaxsat:
                print(" c calling rc2 to find another set of candidates to repair")

                ind = callRC2(
                    maxsatcnfRepair, sigma[2], UniqueVars, Unates, Yvar, YvarOrder
                )

                assert len(ind) > 0

                if args.verbose > 1:
                    print(
                        " c number of candidates undergoing repair iterations", len(ind)
                    )

                if not args.henkin:
                    lexflag, repairfunctions = repair(
                        args,
                        config,
                        repaircnf,
                        ind,
                        Xvar,
                        Yvar,
                        YvarOrder,
                        dg,
                        UniqueVars + Unates,
                        sigma,
                        inputfile_name,
                        args,
                    )
                else:
                    lexflag, repairfunctions = repair(
                        args,
                        config,
                        repaircnf,
                        ind,
                        Xvar,
                        Yvar,
                        YvarOrder,
                        dg,
                        UniqueVars + Unates,
                        sigma,
                        inputfile_name,
                        args,
                        HenkinDep,
                    )

            """
            update the repair candidates in the candidate Skolem
            """

            updateSkolem(repairfunctions, countRefine, sigma[2], inputfile_name, Yvar)

        if countRefine > args.maxrepairitr:
            print(" c number of maximum allowed repair iteration reached")
            print(" c could not synthesize functions")

            break

    end_time = time.time()

    if args.logtime:
        logtime(inputfile_name, "repair time:" + str(end_time - start_time_repair))
        logtime(inputfile_name, "totaltime:" + str(end_time - start_time))
    log_entry.refine_time = end_time - start_time_repair
    log_entry.total_time = end_time - start_time
    log_entry.repair_count = countRefine
    if countRefine == 0:
        log_entry.exit_after_leanskf = True
    else:
        log_entry.exit_after_refine = True
    # log_entry.write_middle_out()


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
        help=("required to enable = 1/disable = 0"
              + "adaptive weighted sampling. Default is 1 "),
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
        help=("num of samples used to learn the candidates."
              + "Takes int value. If not used, Manthan will decide value as per |Y|"),
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
        help=("to learn a subset of existentially quantified variables together"
              + " use --multiclass "),
        action="store_true",
    )
    parser.add_argument(
        "--lexmaxsat",
        help=("to use lexicographical maxsat to find candidates to repair use "
              + "--lexmaxsat "),
        action="store_true",
    )
    parser.add_argument(
        "--henkin",
        help=("if you have dqdimacs instead of qdimacs, "
              + "and would like to learn Henkin functions use --henkin"),
        action="store_true",
    )
    parser.add_argument(
        "--logtime",
        help=("to log time taken in each phase of manthan "
              + "in <inputfile>_timedetails file use --logtime"),
        action="store_true",
    )
    parser.add_argument(
        "--hop",
        help=("if learning candidates via multiclassification, "
              + "hop distances in primal graph "
              + "is used to cluster existentially quantified variables together,"
              + "use --hop <int> to define hop distance. Default is 3"),
        type=int,
        default=3,
        dest="hop",
    )
    parser.add_argument(
        "--clustersize",
        type=int,
        help=("maximum number of existentially quantified variables in a subset"
              + "to be learned together via multiclassfication. Default is 8"),
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
    parser.add_argument("--timeout", type=int, default=7200, dest="timeout")
    parser.add_argument("input", help="input file")

    args = parser.parse_args()

    set_run_pid(args.input)
    atexit.register(log_entry.to_file)
    atexit.register(unset_run_pid, args.input)
    atexit.register(print, " c Manthan exited")

    signal.signal(signal.SIGALRM, handle_timout)
    signal.signal(signal.SIGTERM, handle_exit)

    print(" c configuring dependency paths")
    """
    Paths of dependencies are set.
    if installed by source then their corresponding
    paths are read by "manthan_dependencies.cfg"
    else default is dependencies folder.
    """
    config = configparser.ConfigParser()
    configFilePath = "manthan_dependencies.cfg"
    config.read(configFilePath)

    print(" c starting Manthan")
    mkdir("out")

    try:
        manthan(args, config)
    except AssertionError as e:
        print(" c Manthan crashed")
        print(" c error", e)
        log_entry.exit_after_expection = True
