import ctypes
from typing import Dict

import numpy
from numpy.typing import NDArray

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
