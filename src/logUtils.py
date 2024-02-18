import ctypes
import os
import pickle
import re
from typing import Self

import numpy as np

from src.converToPY import Module, Tokenzier, convert_skf_to_forigen_func


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_from_file(filename):
    with open(filename, "r") as f:
        content = f.read()
    return content


def get_inputfile_contenet(input_file):
    content = get_from_file(input_file)
    # remove enter
    content = content.replace("\n", " ")
    return content


def set_run_pid(input_file):
    path = "run/" + input_file + ".pid"
    with open(path, "w") as f:
        f.write(str(os.getpid()))


def unset_run_pid(input_file):
    path = "run/" + input_file + ".pid"
    os.unlink(path=path)


def to_valid_filename(s):
    # 移除不允许的文件名字符
    s = re.sub(r'[\\/*?:"<>|]', "", s)
    # 也可以考虑将空格替换为下划线
    s = s.replace(" ", "_")
    # 删除其他不合法的字符，根据需要添加
    s = re.sub(r'[^\w.-]', '', s)
    # 防止以系统保留名称命名（例如，在Windows中）
    reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    if s.upper() in reserved_names:
        s = "_" + s
    # 限制文件名长度 (例如，255字符对大多数现代文件系统来说是安全的)
    return s[:255]


class LogEntry:

    def __init__(self) -> None:
        # input
        self.instance_name = ""
        self.instance_str = ""
        self.input_vars = []
        self.output_vars = []
        self.clause_list = []
        self.dag = None
        self.henkin_dep = {}
        # middle out
        self.cnfcontent = None
        self.perprocess_out = None
        self.num_samples = 0
        self.datagen_out = None
        self.leanskf_out = None
        self.errorformula_out = None
        self.maxsatWt = None
        self.maxsatCnf = None
        self.cnfcontent_1 = None
        self.maxsatCnf_1 = list()
        self.cnfcontent_2 = list()
        self.maxsatCnf_2 = list()
        self.cex_model = list()
        self.indlist = list()
        self.repair_loop = list()
        self.repaired_func = list()
        # time
        self.total_time = 0
        self.preprocess_time = 0
        self.datagen_time = 0
        self.leanskf_time = 0
        self.refine_time = 0
        self.repair_count = 0
        # output
        self.output_verilog = ""
        self.circuit_size = -1
        self.sat = False
        # misc
        self.exit_after_preprocess = False
        self.exit_after_leanskf = False
        self.exit_after_refine = False
        self.exit_after_error = False
        self.exit_after_timeout = False
        self.exit_after_expection = False
        self.exit_at_progress = False

    def caculate_circuit_size(self, recaculate=False):
        if not recaculate and self.circuit_size != -1:
            return self.circuit_size

        if self.output_verilog == "":
            self.circuit_size = 0
            return self.circuit_size

        tokens = list()
        for token in Tokenzier(self.output_verilog):
            tokens.append(token)

        module = Module(tokens)
        self.circuit_size = len(module.exprs)
        return self.circuit_size

    def get_samples_acc(self):
        if self.num_samples == 0:
            return 0, 0

        if self.output_verilog == "":
            return 0, self.num_samples
        safe_instane_name = to_valid_filename(self.instance_name)
        func = convert_skf_to_forigen_func(self.output_verilog, safe_instane_name)
        acc = 0
        for input in self.datagen_out:
            input = [bool(item) for item in input]
            args = (ctypes.c_bool * len(input))(*input)
            ret = func(args)
            output = list()
            for i in range(len(input)):
                output.append(ret[i])
            # compare input and output is the same
            if np.array_equal(input, output):
                acc += 1

        return acc, self.num_samples

    def get_clause_list(self):
        self.clause_list = []
        lines = self.instance_str.split(" 0 ")
        for line in lines:
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                continue
            if line.startswith("a"):
                continue
            if line.startswith("e"):
                continue
            if line.startswith("d"):
                continue
            clause = line.strip(" ").strip("\n").strip(" ").split(" ")[:-1]
            if len(clause) > 0:
                clause = list(map(int, list(clause)))
                self.clause_list.append(clause)

        return self.clause_list

    def get_clause_list_avg_len(self):
        clause_list = self.get_clause_list()
        if len(clause_list) == 0:
            return 0

        total_len = 0
        for clause in clause_list:
            total_len += len(clause)

        return total_len / len(clause_list)

    @staticmethod
    def from_file(filename) -> Self:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def to_file(self):
        if self.output_verilog != "":
            self.sat = True
        mkdir("log")
        path = "log/" + str(self.instance_name) + ".pkl"
        try:
            f = open(path, "wb")
            pickle.dump(self, f)
        except OSError:
            print("OSError: " + path)
        finally:
            f.close()

    def write_middle_out(self):
        mkdir("log-middle")

        cnfcontent_out_path = "log-middle/" + str(
            self.instance_name) + ".cnfcontent"
        preprocess_out_path = "log-middle/" + str(
            self.instance_name) + ".preprocess"
        datagen_out_path = "log-middle/" + str(self.instance_name) + ".datagen"
        leanskf_out_path = "log-middle/" + str(self.instance_name) + ".leanskf"
        errorformula_out_path = ("log-middle/" + str(self.instance_name) +
                                 ".errorformula")

        with open(cnfcontent_out_path, "w") as f:
            f.write(self.cnfcontent)

        with open(preprocess_out_path, "w") as f:
            f.write(str(self.perprocess_out))

        with open(datagen_out_path, "w") as f:
            for i in range(self.num_samples):
                f.write(np.array2string(self.datagen_out[i]))
                f.write("\n")

        with open(leanskf_out_path, "w") as f:
            f.write(self.leanskf_out)

        with open(errorformula_out_path, "w") as f:
            f.write(self.errorformula_out)
