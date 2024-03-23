import ctypes
import random
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy
from numpy.typing import NDArray

from src.candidateSkolem import createCluster
from src.converToPY import convert_skf_to_forigen_func


def fullsample(res_verilog: str, datagen_out: NDArray) -> NDArray:
    # use res verilog to build func
    func, var_order = convert_skf_to_forigen_func(res_verilog)

    # convert rich var_order info to index
    # func param index -> sample index
    # 0: 2 means the first parm in func is mapping to index 2(the third one) in smaple
    mapping_dict: Dict = {}
    has_out: bool = False
    out_idx: int = 0
    max_sample_index: int = -1
    for idx, var in enumerate(var_order):
        # normally the var is a str like o114514, so i can get index use [1:]
        # if  found the var is `out`, keep it to later process.
        if var == "out":
            has_out = True
            out_idx = idx
            continue
        sample_index = int(var[1:]) - 1
        max_sample_index = max(max_sample_index, sample_index)
        mapping_dict[idx] = sample_index

    if has_out:
        sample_index = max_sample_index + 1
        mapping_dict[out_idx] = sample_index
        # check sample length
        assert (
            sample_index + 1 <= datagen_out.shape[1]
        ), "the sample is shorter than required"

    # make a full sample
    res_sample_list = list()
    for sample in datagen_out:
        # collect input fromm sample
        input = [sample[mapping_dict[param_idx]] for param_idx in range(len(var_order))]
        # convert it to bool
        input = list(map(bool, input))
        args = (ctypes.c_bool * len(input))(*input)
        output = func(args)
        for idx in range(len(input)):
            sample[mapping_dict[idx]] = int(output[idx])

        res_sample_list.append(sample)

    return numpy.array(res_sample_list)


def genRandomVerilog(target: int, Yset: List, Xvar: List) -> Tuple[str, List]:
    # firstly, we have a list of symbols
    # xvar
    symbols_list = ["i" + str(var) for var in Xvar]
    # relative yvar
    symbols_list.extend(["w" + str(var) for var in Yset if var != target])
    # constant
    symbols_list.extend(["0", "1"])
    random.shuffle(symbols_list)

    MAGICAL_NUM: int = int(len(symbols_list) / 2 + 1)

    # Now we get random clause num
    clause_num: int = random.randint(1, MAGICAL_NUM)
    conjoinclause_list: List = list()
    depend_list: Set = set()
    for _ in range(clause_num):
        # choose symbol
        left_part, right_part = random.choices(symbols_list, k=2)

        # build dependency list
        if left_part[0] == "w":
            depend_list.add(int(left_part[1:]))
        if right_part[0] == "w":
            depend_list.add(int(right_part[1:]))

        # is negative
        if (random.randrange(1, 114514) - (114514 / 2)) < 0:
            if len(left_part) == 1:
                left_part = str(1 - int(left_part))
            else:
                left_part = "~" + left_part
        if (random.randrange(1, 114514) - (114514 / 2)) > 0:
            if len(right_part) == 1:
                right_part = str(1 - int(right_part))
            else:
                right_part = "~" + right_part

        clause = f"( {left_part} & {right_part} )"
        # is clause is negative
        if (random.randrange(1, 114514) - (114514 / 2)) > 0:
            clasue = f"(~{clause})"

        conjoinclause_list.append(clause)

    res_expr = "|".join(conjoinclause_list)
    res_expr = "(" + res_expr + ")"

    return res_expr, list(depend_list)


def generateRandomSkf(Yset: List, Xvar: List) -> Tuple[Dict, Dict]:
    psi_dict: Dict = {}
    D_dict: Dict[int, List] = {}
    # do random generate
    for yvar in Yset:
        psi_dict[yvar], D_dict[yvar] = genRandomVerilog(yvar, Yset, Xvar)

    return psi_dict, D_dict


def learnRandomCandidate(
    Xvar, Yvar, UniqueVars, PosUnate, NegUnate, samples, dg, ng, args, HenkinDep={}
):
    candidateSkf = (
        {}
    )  # represents y_i and its corresponding learned candidate via decision tree.

    SkolemKnown = PosUnate + NegUnate + UniqueVars

    for var in SkolemKnown:
        if (args.multiclass) and (var in list(ng.nodes)):
            ng.remove_node(var)

        for var in PosUnate:
            candidateSkf[var] = " 1 "

        for var in NegUnate:
            candidateSkf[var] = " 0 "

    disjointSet = createCluster(args, Yvar, SkolemKnown, ng)

    for Yset in disjointSet:
        dependent = []
        for yvar in Yset:
            if not args.henkin:
                depends_on_yvar = list(nx.ancestors(dg, yvar))
                depends_on_yvar.append(yvar)
                dependent = dependent + depends_on_yvar
            else:
                yvar_depends_on = list(nx.descendants(dg, yvar))
                if yvar in list(yvar_depends_on):
                    yvar_depends_on.remove(yvar)

        functions, D_set = generateRandomSkf(Yset, Xvar)

        for var in Yset:
            assert var not in UniqueVars
            assert var not in PosUnate
            assert var not in NegUnate
            candidateSkf[var] = functions[var]
            D = list(set(D_set[var]) - set(Xvar))
            for jvar in D:
                dg.add_edge(var, jvar)

    if args.verbose:
        print(" c generated candidate functions for all variables.")

    if args.verbose == 2:
        print(" c candidate functions are", candidateSkf)

    return candidateSkf, dg
