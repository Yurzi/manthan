import pickle
import os
import numpy as np
import subprocess as subp
from typing import Self
from src.converToPY import convert_skf_to_pyfunc, repair_skf_verilog, Module, Tokenzier


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
        f.write("running")


def unset_run_pid(input_file):
    path = "run/" + input_file + ".pid"
    os.unlink(path=path)


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
            return 0

        if self.output_verilog == "":
            return 0, self.num_samples

        func = convert_skf_to_pyfunc(self.output_verilog)
        acc = 0
        for input in self.datagen_out:
            input = [bool(item) for item in input]
            output = func(*input)
            acc_flag = True
            for i in range(0, len(input)):
                if input[i] == output[i]:
                    continue
                else:
                    acc_flag = False
                    break
            if acc_flag:
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
        mkdir("log")
        path = "log/" + str(self.instance_name) + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        f.close()

    def write_middle_out(self):
        mkdir("log-middle")

        cnfcontent_out_path = "log-middle/" + str(self.instance_name) + ".cnfcontent"
        preprocess_out_path = "log-middle/" + str(self.instance_name) + ".preprocess"
        datagen_out_path = "log-middle/" + str(self.instance_name) + ".datagen"
        leanskf_out_path = "log-middle/" + str(self.instance_name) + ".leanskf"
        errorformula_out_path = (
            "log-middle/" + str(self.instance_name) + ".errorformula"
        )

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
