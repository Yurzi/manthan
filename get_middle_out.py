from src.logUtils import LogEntry
import pprint

log_entry = LogEntry.from_file("log/qbffam_KBKF_QU_2.pkl")

print("-------- Candidates SKF --------")
print(log_entry.leanskf_out)
print("\n")
print("-------- Error Formula --------")
print(log_entry.errorformula_out)
print("\n")
print("-------- Cnf Content --------")
print(log_entry.cnfcontent)
print("\n")
print("-------- Maxsat Wt --------")
print(log_entry.maxsatWt)
print("\n")
print("-------- Maxsat CNF --------")
print(log_entry.maxsatCnf)
print("\n")
print("-------- Cnf Content 1 --------")
print(log_entry.cnfcontent_1)
print("\n")

repire_count = len(log_entry.repair_loop)

print("Repair Count: %d" % repire_count)

for idx in range(repire_count):
    print("======== Repair Loop %d ========" % idx)
    print("------ Cex Model ------")
    print(log_entry.cex_model[idx])
    print("------ Maxsat CNF 1 (MaxsatCNFrepair) ------")
    print(log_entry.maxsatCnf_1[idx])
    print("------ Cnf Content 2 (Repair CNF) ------")
    print(log_entry.cnfcontent_2[idx])
    print("------ Maxsat CNF 2 (MaxCNF) ------")
    print(log_entry.maxsatCnf_2[idx])
    print("------ Indlist ------")
    print(log_entry.indlist[idx])
    print("------ In Repair------")
    loop_entry = log_entry.repair_loop[idx]
    for i, entry in enumerate(loop_entry):
        print(".... In Repair %d ...." % i)
        pprint.pprint(entry)

    print("------- Repaired Func -------")
    print(log_entry.repaired_func[idx])
    print("\n")



