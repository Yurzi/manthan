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
    # misc
    self.exit_after_preprocess = False
    self.exit_after_leanskf = False
    self.exit_after_refine = False
  
  def caculate_circuit_size(self, recaculate=False):
    if recaculate or self.circuit_size == -1:
      self.circuit_size = 0 # TODO: calculate circuit size
    return self.circuit_size
  
def get_from_file(filename):
  with open(filename,"r") as f:
    content = f.read()
  f.close()
  return content

def get_inputfile_contenet(input_file):
  content = get_from_file(input_file)
  # remove enter
  content = content.replace("\n"," ")
  return content