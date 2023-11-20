import pickle
import os
import numpy as np
import subprocess as subp
from typing import Self
from src.converToPY import convert_skf_to_pyfunc


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

        # caculate circuit size use external tool
        # 1. write verilog to file
        mkdir("run")
        verilog_path = "run/" + str(self.instance_name) + ".v"
        with open(verilog_path, "w") as f:
            f.write(self.output_verilog)
        f.close()
        # 2. wirte script to file
        script_path = "run/" + str(self.instance_name) + ".script"
        with open(script_path, "w") as f:
            f.write("read_verilog " + verilog_path + "\n")
            f.write("synth -top SkolemFormula -flatten\n")
            f.write("abc -g NAND\n")
            f.write("stat")
        f.close()
        # 3. run script
        executor = "./oss-cad-suite/bin/yosys"
        cmd = [executor, "-s", script_path]
        p = subp.Popen(cmd, stdout=subp.PIPE, stderr=subp.PIPE)
        out, err = p.communicate()
        # 4. parse output
        temp = []
        out = out.decode("utf-8")
        out = out.split("\n")
        for line in out:
            if "Number of cells" in line:
                temp.append(int(line.split(" ")[-1]))

        self.circuit_size = temp[-1]
        return self.circuit_size
    
    def get_samples_acc(self) -> float:
        if self.num_samples == 0:
            return 0
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

        return acc / self.num_samples

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
