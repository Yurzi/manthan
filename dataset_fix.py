import argparse
import os


def parse_qdimacs(qdimacs_str):
    # parse to plain string
    Xvar = []
    Yvar = []
    qdimacs_list = []

    HenkinDep = {
    }  # for DQBF: it presents variables with explict dependencies.

    lines = qdimacs_str.split("\n")
    for line in lines:
        if line.startswith("c"):
            continue
        if (line == "") or (line == "\n"):
            continue

        if line.startswith("p"):
            continue

        if line.startswith("a"):
            Xvar += line.strip("a").strip("\n").strip(" ").split(" ")[:-1]
            continue

        if line.startswith("e"):
            Yvar += line.strip("e").strip("\n").strip(" ").split(" ")[:-1]
            continue

        if line.startswith("d"):
            YDep = line.strip("d").strip("\n").strip(" ").split(" ")[:-1]
            dvar = int(YDep[0])
            Yvar.append(dvar)
            HenkinDep[dvar] = list(map(int, list(YDep[1:])))
            continue

        clause = line.strip(" ").strip("\n").strip(" ").split(" ")[:-1]

        if len(clause) > 0:
            clause = list(map(int, list(clause)))
            qdimacs_list.append(clause)

    if (len(Xvar) == 0) or (len(Yvar) == 0) or (len(qdimacs_list) == 0):
        print(" c problem with the files, can not synthesis Skolem functions")
        raise SystemExit()

    Xvar = list(map(int, Xvar))
    Yvar = list(map(int, Yvar))

    Xvar = sorted(Xvar)
    Yvar = sorted(Yvar)

    return Xvar, Yvar, qdimacs_list


def get_files(dir: str):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".qdimacs"):
                yield os.path.join(root, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", help="datatset dir", default="dataset")
    args = parser.parse_args()

    dir_path = args.dir_path
    for file in get_files(dir_path):
        # parse
        qdimacs_str = ""

        with open(file, "r") as f:
            qdimacs_str = f.read()
            Xvar, Yvar, qdimacs_list = parse_qdimacs(qdimacs_str)
            Vars = Xvar + Yvar

        qdimacs_lines = qdimacs_str.splitlines()
        define_stm = ""
        define_stm_idx = -1
        for idx, line in enumerate(qdimacs_lines):
            if line.startswith("p"):
                define_stm = line
                define_stm_idx = idx
                break
        
        tokens = define_stm.split(" ")
        var_num = int(tokens[2])
        clause_num = int(tokens[3])

        need_fix = False

        if len(qdimacs_list) != clause_num:
            print(f"Error: {file} clause_num not match")
            need_fix = True
        if max(Vars) != var_num:
            print(f"Error: {file} var_num not match")
            need_fix = True

        if need_fix:
            print(f"Fixing: {file}")
            qdimacs_lines[define_stm_idx] = f"p cnf {len(Xvar) + len(Yvar)} {len(qdimacs_list)}"
            with open(file, "w") as f:
                f.write("\n".join(qdimacs_lines))
