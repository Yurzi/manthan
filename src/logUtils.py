import pickle
import os
import numpy as np

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
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
  
  def __str__(self):
      return json.dumps(dict(self), ensure_ascii=False)
  
  
  def caculate_circuit_size(self, recaculate=False):
    if recaculate or self.circuit_size == -1:
      self.circuit_size = 0 # TODO: calculate circuit size
    return self.circuit_size
  
  
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
    errorformula_out_path = "log-middle/" + str(self.instance_name) + ".errorformula"
  
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
    
  
def get_from_file(filename):
  with open(filename,"r") as f:
    content = f.read()
  return content


def get_inputfile_contenet(input_file):
  content = get_from_file(input_file)
  # remove enter
  content = content.replace("\n"," ")
  return content

def set_run_pid(input_file):
  path = "run/" + input_file + ".pid"
  with open(path, "w") as f:
    f.write("running")

def unset_run_pid(input_file):
  path = "run/" + input_file + ".pid"
  os.unlink(path=path)