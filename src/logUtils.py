import os
import pickle
import re
from typing import Any, Dict, List, Self, Optional

from networkx.classes import DiGraph
import numpy as np
from numpy._typing import NDArray

from src.converToPY import Module, Tokenzier, get_verilog_input_order


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_from_file(filename: os.PathLike | str) -> str:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File: {filename} not found")
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
    reserved_names = {
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9"
    }
    if s.upper() in reserved_names:
        s = "_" + s
    # 限制文件名长度 (例如，255字符对大多数现代文件系统来说是安全的)
    return s[:255]


def check_and_reset(filepath, is_dir: bool = False):
    if is_dir:
        if os.path.exists(filepath):
            os.rmdir(filepath)
        os.mkdir(filepath)
    else:
        if os.path.exists(filepath):
            os.remove(filepath)


class LogEntry:

    def __init__(self) -> None:
        # input
        self.instance_name: str = ""
        self.instance_str: str = ""
        self.input_vars: List = []
        self.output_vars: List = []
        self.clause_list: List = []
        self.dag: Optional[DiGraph] = None
        self.henkin_dep: Dict = {}
        # middle out
        self.cnfcontent: Optional[str] = None
        self.perprocess_out: Optional[List] = None
        self.num_samples: int = 0
        self.datagen_out: Optional[NDArray] = None
        self.leanskf_out: Optional[str] = None
        self.errorformula_out: Optional[str] = None
        self.maxsatWt: Optional[int] = None
        self.maxsatCnf: Optional[str] = None
        self.cnfcontent_1: Optional[str] = None
        self.maxsatCnf_1 = list()
        self.cnfcontent_2 = list()
        self.maxsatCnf_2 = list()
        self.cex_model = list()
        self.indlist = list()
        self.repair_loop = list()
        self.repaired_func = list()
        # time
        self.total_time: float = 0.0
        self.preprocess_time: float = 0.0
        self.datagen_time: float = 0.0
        self.leanskf_time: float = 0.0
        self.refine_time: float = 0.0
        self.repair_count: int = 0
        # output
        self.output_verilog: str = ""
        self.circuit_size: float = -1
        self.sat: bool = False
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

        if self.leanskf_out == "" or self.leanskf_out is None:
            return 0, self.num_samples
        safe_instane_name = to_valid_filename(self.instance_name)
        # write leanskf out to file
        leanskf_out_path = "run/" + safe_instane_name + "_learnskf" + ".v"
        var_order = get_verilog_input_order(self.leanskf_out)
        with open(leanskf_out_path, "w") as f:
            f.write(self.leanskf_out)

        # use file_generate_cnf to generate cnf file
        cnf_file_path = "run/" + safe_instane_name + "_learnskf" + ".cnf"
        mapping_file_path = "run/" + safe_instane_name + "_learnskf" + ".map"
        file_generate_cnf_exe = "dependencies/static_bin/file_generation_cnf"
        file_generate_cnf_cmd = (file_generate_cnf_exe + " " +
                                 leanskf_out_path + " " + cnf_file_path + " " +
                                 mapping_file_path)
        ret = os.system(file_generate_cnf_cmd)
        if ret != 0:
            print("    error in file_generate_cnf_cmd")
            return -1, self.num_samples

        # read cnf file
        cnf_content = ""
        with open(cnf_file_path, "r") as f:
            cnf_content = f.read()

        # read mapping file
        mapping_content = ""
        with open(mapping_file_path, "r") as f:
            mapping_content = f.read()

        # combine mapping with var_order
        var_mapping_v_to_abccnf = dict()
        var_mapping_cnf_to_v = dict()
        mapping_content = mapping_content.split(" ")[:-2]
        mapping_content = list(map(int, mapping_content))
        for i in range(len(var_order)):
            var_mapping_v_to_abccnf[var_order[i]] = mapping_content[i]
            var = var_order[i]
            idx = var[1:]
            var_mapping_cnf_to_v[int(idx)] = var

        # use pysat to solve cnf
        from pysat.formula import CNF
        from pysat.solvers import Solver
        cnf = CNF()
        cnf.from_string(cnf_content)
        acc = 0
        with Solver(bootstrap_with=cnf) as solver:
            for idx, sample in enumerate(self.datagen_out):
                assumption = [2]
                for i in range(len(sample)):
                    verliog_var = var_mapping_cnf_to_v.get(i + 1, None)
                    if verliog_var is None:
                        continue
                    abc_cnf_var = var_mapping_v_to_abccnf[verliog_var]

                    if sample[i] == 0:
                        assumption.append(-abc_cnf_var)
                    else:
                        assumption.append(abc_cnf_var)
                res = solver.solve(assumptions=assumption)

                if res:
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
