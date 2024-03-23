import ctypes
import os
from enum import Enum
from typing import Callable, List, Optional, Self, Tuple
import re
import networkx as nx


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


class Token:

    class Kind(Enum):
        Keyword = "Keyword"
        Ident = "Ident"
        Literal = "Literal"

        And = "And"
        Or = "Or"
        Not = "Not"
        Xor = "Xor"
        Paren = "Paren"
        Eq = "Eq"

        Whitespace = "Whitespace"
        Comma = "Comma"
        SemiColon = "SemiColon"
        Enter = "Enter"
        Unknown = "Unknown"
        Eof = "Eof"

    def __init__(self, kind: Kind, lexme: str) -> None:
        self.kind: Token.Kind = kind
        self.lexme: str = lexme

    def __str__(self) -> str:
        res = f"Token: Kind:{self.kind}\tLexme:{self.lexme}"
        return res


class Tokenzier:

    class Cursor:
        EOF_CHAR = "\0"

        def __init__(self, input: str) -> None:
            self._chars: str = input
            self._index: int = 0
            self._mem: str = ""

        @property
        def mem(self) -> str:
            return self._mem

        def reset_mem(self) -> None:
            self._mem = ""

        @property
        def is_eof(self) -> bool:
            if self._index >= len(self._chars):
                return True

            return False

        @property
        def first(self) -> str:
            if self._index >= len(self._chars):
                return self.EOF_CHAR

            return self._chars[self._index]

        @property
        def second(self) -> str:
            if self._index + 1 >= len(self._chars):
                return self.EOF_CHAR

            return self._chars[self._index + 1]

        def bump(self) -> str | None:
            if self._index >= len(self._chars):
                return None
            c = self._chars[self._index]
            self._index += 1

            self._mem += c
            return c

        def eat_while(self, predicate: Callable) -> None:
            while predicate(self.first) and not self.is_eof:
                self.bump()

    def __init__(self, input: str) -> None:
        self.raw_str: str = input
        self.cursor: Tokenzier.Cursor = self.Cursor(input)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Token:
        token = self._advance_token()
        if token.kind == Token.Kind.Eof:
            raise StopIteration
        return token

    @staticmethod
    def is_whitespace(c: str) -> bool:
        if c == " ":
            return True
        if c == "\t":
            return True
        return False

    @staticmethod
    def is_enter(c: str) -> bool:
        if c == "\n":
            return True
        if c == "\r":
            return True

        return False

    @staticmethod
    def is_id_start(c: str) -> bool:
        return c.isalpha()

    @staticmethod
    def is_id_contine(c: str) -> bool:
        return c.isalnum() or c == "_"

    @staticmethod
    def is_lit_start(c: str) -> bool:
        return c.isdigit() or c == "'"

    @staticmethod
    def is_lit_part(c: str) -> bool:
        return c.isdigit(
        ) or c == "_" or c == "b" or c == "o" or c == "h" or c == "d" or c == "'"

    @staticmethod
    def is_lit_contine(c: str) -> bool:
        return c.isdigit() or c == "_"

    def lit(self) -> None:
        now_char = self.cursor.first  # eat ' or a digit or b or o or h
        if not self.is_lit_part(now_char):
            return
        if now_char == "'":
            self.cursor.bump()  # eat '
            self.cursor.bump()  # eat b or o or h or d
            self.cursor.eat_while(self.is_lit_contine)
            return
        if now_char == "b" or now_char == "o" or now_char == "h" or now_char == "d":
            self.cursor.bump()  # eat b or o or h or d
            self.cursor.eat_while(self.is_lit_contine)
            return
        if now_char.isdigit():
            self.cursor.eat_while(self.is_lit_contine)
            next_char = self.cursor.first
            if next_char == "'":
                self.cursor.bump()  # eat '
                self.cursor.bump()  # eat b or o or h or d
                self.cursor.eat_while(self.is_lit_contine)  # eat lit
                return
            return
        return

    @staticmethod
    def is_keyword(lexme: str) -> bool:
        if lexme in [
                "module", "input", "output", "wire", "assign", "endmodule"
        ]:
            return True
        return False

    def _advance_token(self) -> Token:
        first_char = self.cursor.bump()
        if first_char is None:
            return Token(Token.Kind.Eof, "\0")

        token_kind = Token.Kind.Unknown
        if self.is_whitespace(first_char):
            self.cursor.eat_while(self.is_whitespace)
            token_kind = Token.Kind.Whitespace
        if self.is_id_start(first_char):
            self.cursor.eat_while(self.is_id_contine)
            token_kind = Token.Kind.Ident
        if self.is_lit_start(first_char):
            self.lit()
            token_kind = Token.Kind.Literal
        if self.is_enter(first_char):
            self.cursor.eat_while(self.is_enter)
            token_kind = Token.Kind.Enter
        if first_char == "&":
            token_kind = Token.Kind.And
        if first_char == "~":
            token_kind = Token.Kind.Not
        if first_char == "|":
            token_kind = Token.Kind.Or
        if first_char == "^":
            token_kind = Token.Kind.Xor
        if first_char == "=":
            token_kind = Token.Kind.Eq
        if first_char == "(" or first_char == ")":
            token_kind = Token.Kind.Paren
        if first_char == ",":
            token_kind = Token.Kind.Comma
        if first_char == ";":
            token_kind = Token.Kind.SemiColon
        token = Token(token_kind, self.cursor.mem)
        self.cursor.reset_mem()
        return self.refine(token)

    @staticmethod
    def refine(token: Token) -> Token:
        if token.kind is Token.Kind.Ident:
            if Tokenzier.is_keyword(token.lexme):
                token.kind = Token.Kind.Keyword

        return token


class StmLine:

    class Kind(Enum):
        Module = "module"
        Input = "input"
        Output = "output"
        Wire = "wire"
        Assign = "assign"
        Unknown = "unknown"

    def __init__(self, lineno: int, tokens: list[Token]) -> None:
        self.kind: StmLine.Kind = StmLine.Kind.Unknown
        self.lineno = lineno
        self.tokens = tokens
        self.parse()

    def append(self, token: Token) -> None:
        self.tokens.append(token)

    def parse(self) -> None:
        res: list[Token] = list()
        for token in self.tokens:
            if token.kind is Token.Kind.Whitespace:
                continue
            if token.kind is Token.Kind.Enter:
                continue
            if token.kind is Token.Kind.SemiColon:
                continue
            if token.kind is Token.Kind.Comma:
                continue

            res.append(token)
        keyword = res[0].lexme
        if keyword == "module":
            self.kind = self.Kind.Module
        if keyword == "input":
            self.kind = self.Kind.Input
        if keyword == "output":
            self.kind = self.Kind.Output
        if keyword == "wire":
            self.kind = self.Kind.Wire
        if keyword == "assign":
            self.kind = self.Kind.Assign

        self.tokens = res

    def __str__(self) -> str:
        res = f"StmLine: Line: {self.lineno}  Kind: {self.kind} Tokens:\n"
        for token in self.tokens:
            res += str(token) + "\n"

        return res

    @staticmethod
    def from_tokens(tokens: list[Token]) -> list['StmLine']:
        lineno: int = 1
        res: list[StmLine] = list()
        line: list[Token] = list()
        for token in tokens:
            if token.kind is Token.Kind.Enter:
                res.append(StmLine(lineno, line))
                lineno += 1
                line = list()
                continue

            line.append(token)

        return res


class Expression:

    def __init__(self, stm: StmLine) -> None:
        self.stm = stm
        self.expr: list[Token] = list()
        self.income: list[Token] = list()
        self.outcome: Token = Token(Token.Kind.Ident, "unknown")
        self.parse()

    def parse(self) -> None:
        expr = list()
        tokens = self.stm.tokens
        # remove keyword
        if tokens[0].kind is not Token.Kind.Keyword:
            raise Exception("Invalid expression")

        for index in range(1, len(tokens)):
            token = tokens[index]
            expr.append(token)

            if index == 1:
                self.outcome = token
                continue
            if index == 2 and token.kind is Token.Kind.Eq:
                continue

            if token.kind is Token.Kind.Ident or token.kind is Token.Kind.Literal:
                self.income.append(token)

        self.expr = expr

    def gen_pycode(self) -> str:
        can_add_lit = True
        res = []
        for token in self.expr:
            if token.kind is Token.Kind.And:
                res.append("and")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Or:
                res.append("or")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Not:
                res.append("not")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Xor:
                res.append("!=")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Paren:
                res.append(token.lexme)
                can_add_lit = False
                continue
            if token.kind is Token.Kind.Eq:
                res.append("=")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Literal:
                if can_add_lit is False:
                    continue
                literal = token.lexme.split("'")
                if len(literal) > 1:
                    literal = literal[-1][1:]
                else:
                    literal = literal[0]

                if int(literal) >= 1:
                    res.append("True")
                else:
                    res.append("False")
                can_add_lit = False
                continue

            res.append(token.lexme)
            can_add_lit = False
        res = " ".join(res)
        return res

    def gen_cppcode(self) -> str:
        can_add_lit = True
        res = []
        for token in self.expr:
            if token.kind is Token.Kind.And:
                res.append("&&")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Or:
                res.append("||")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Not:
                res.append("!")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Xor:
                res.append("!=")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Paren:
                res.append(token.lexme)
                can_add_lit = False
                continue
            if token.kind is Token.Kind.Eq:
                res.append("=")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Literal:
                if can_add_lit is False:
                    continue
                literal = token.lexme.split("'")

                if len(literal) > 1:
                    literal = literal[-1][1:]
                else:
                    literal = literal[0]

                if int(literal, 2) == 0:
                    res.append("false")
                elif int(literal, 2) == 1:
                    res.append("true")
                else:
                    res.append(str(int(literal, 2)))
                can_add_lit = False
                continue

            res.append(token.lexme)
            can_add_lit = False
        res = " ".join(res)
        return res

    def gen_verilog(self) -> str:
        can_add_lit = True
        res = []
        for token in self.expr:
            if token.kind is Token.Kind.And:
                res.append("&")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Or:
                res.append("|")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Not:
                res.append("~")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Xor:
                res.append("^")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Paren:
                res.append(token.lexme)
                can_add_lit = False
                continue
            if token.kind is Token.Kind.Eq:
                res.append("=")
                can_add_lit = True
                continue
            if token.kind is Token.Kind.Literal:
                if can_add_lit is False:
                    continue
                res.append(token.lexme)
                can_add_lit = False
                continue

            res.append(token.lexme)
            can_add_lit = False
        res = " ".join(res)
        return res

    def __str__(self) -> str:
        res = "Expr:\n"
        res += "Income:\n"
        for token in self.income:
            res += str(token) + "\n"
        res += "Outcome:\n" + str(self.outcome) + "\nExprs:\n"
        for token in self.expr:
            res += str(token) + "\n"

        return res


class Module:

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.name: Token = Token(Token.Kind.Ident, "SkolemFormula")
        self.input_vars: list[Token] = list()
        self.output_vars: list[Token] = list()
        self.inner_vars: list[Token] = list()
        self.exprs: list[Expression] = list()
        self.parse()

    def parse(self) -> None:
        stms = StmLine.from_tokens(self.tokens)

        for stm in stms:
            if stm.kind is StmLine.Kind.Module:
                self.name = stm.tokens[1]
            if stm.kind is StmLine.Kind.Input:
                self.input_vars.extend(stm.tokens[1:])
            if stm.kind is StmLine.Kind.Output:
                self.output_vars.extend(stm.tokens[1:])
            if stm.kind is StmLine.Kind.Wire:
                self.inner_vars.extend(stm.tokens[1:])

            if stm.kind is StmLine.Kind.Assign:
                self.exprs.append(Expression(stm))

    def reorgnize(self) -> None:
        vertex = list()
        tmp = list()
        tmp.extend(self.input_vars)
        tmp.extend(self.inner_vars)
        tmp.extend(self.output_vars)

        for token in tmp:
            vertex.append(token.lexme)

        graph = nx.DiGraph()
        graph.add_nodes_from(vertex)
        for expr in self.exprs:
            for token in expr.income:
                if token.kind is Token.Kind.Literal:
                    continue

                graph.add_edge(token.lexme, expr.outcome.lexme)

        sorted_vars = list(nx.topological_sort(graph))
        sorted_exprs = list()
        for expr_out in sorted_vars:
            for expr in self.exprs:
                if expr.outcome.lexme == expr_out:
                    sorted_exprs.append(expr)
        assert len(sorted_exprs) == len(
            self.exprs), "some experssion is missing"
        self.exprs = sorted_exprs

    def sort_input(self):
        input_vars = [var.lexme for var in self.input_vars]

        def special_sort(vars: list) -> list:
            for i in range(1, len(vars)):
                for j in range(0, len(vars) - i):
                    if vars[j] == "out":
                        vars[j], vars[j + 1] = vars[j + 1], vars[j]
                        continue

                    if vars[j + 1] == "out":
                        continue

                    var_1 = int(vars[j][1:])
                    var_2 = int(vars[j + 1][1:])

                    if var_1 > var_2:
                        vars[j], vars[j + 1] = vars[j + 1], vars[j]
            return vars

        input_vars = special_sort(input_vars)
        res = list()
        for var in input_vars:
            for var_token in self.input_vars:
                if var_token.lexme == var:
                    res.append(var_token)
                    break

        self.input_vars = res

    def gen_pycode(self) -> str:

        def special_sort(vars: list) -> list:
            for i in range(1, len(vars)):
                for j in range(0, len(vars) - i):
                    if vars[j] == "out":
                        vars[j], vars[j + 1] = vars[j + 1], vars[j]
                        continue
                    if vars[j + 1] == "out":
                        continue

                    var_1 = int(vars[j][1:])
                    var_2 = int(vars[j + 1][1:])

                    if var_1 > var_2:
                        vars[j], vars[j + 1] = vars[j + 1], vars[j]
            return vars

        res = list()
        # func def
        var_list = [var.lexme for var in self.input_vars]
        var_list.extend([var.lexme for var in self.output_vars])
        var_list = special_sort(var_list)
        var_list.append("*args")
        var_list = ", ".join(var_list)

        res.append(f"def {self.name.lexme}({var_list}):")
        # expr
        exprs_def = list()
        for expr in self.exprs:
            exprs_def.append("    " + expr.gen_pycode())

        res.extend(exprs_def)

        # return
        ret_vars = [var.lexme for var in self.output_vars]
        ret_vars.extend([var.lexme for var in self.input_vars])
        ret_vars = special_sort(ret_vars)
        if len(ret_vars) > 0:
            ret_vars.append("*args")
        else:
            ret_vars.append("args")

        ret_vars = ", ".join(ret_vars)
        ret = f"    return {ret_vars}"
        res.append(ret)

        res = "\n".join(res)
        return res

    def gen_cppcode(self) -> Tuple[str, List]:

        def special_sort(vars: list) -> list:
            for i in range(1, len(vars)):
                for j in range(0, len(vars) - i):
                    if vars[j] == "out":
                        vars[j], vars[j + 1] = vars[j + 1], vars[j]
                        continue
                    if vars[j + 1] == "out":
                        continue

                    var_1 = int(vars[j][1:])
                    var_2 = int(vars[j + 1][1:])

                    if var_1 > var_2:
                        vars[j], vars[j + 1] = vars[j + 1], vars[j]
            return vars

        res = list()
        # func def
        var_list = [var.lexme for var in self.input_vars]
        var_list.extend([var.lexme for var in self.output_vars])
        var_list = special_sort(var_list)
        var_order = var_list

        res.append("bool* skf_func(bool* args)")
        res.append("{")

        for i in range(len(var_list)):
            res.append("    bool " + var_list[i] + " = args[" + str(i) + "];")
        # allocate memory for return
        ret_vars = [var.lexme for var in self.input_vars]
        ret_vars.extend([var.lexme for var in self.output_vars])
        ret_vars = special_sort(ret_vars)

        malloc_stm = f"    bool* ret = (bool*)malloc({len(ret_vars)} * sizeof(bool));"
        memset_stm = f"    memset(ret, 0, {len(ret_vars)} * sizeof(bool));"
        res.append(malloc_stm)
        res.append(memset_stm)

        # expr
        exprs_def = list()
        for var in self.inner_vars:
            exprs_def.append(f"    bool {var.lexme};")
        for expr in self.exprs:
            exprs_def.append("    " + expr.gen_cppcode() + ";")

        res.extend(exprs_def)

        # return
        ret_stms = list()
        for i in range(len(ret_vars)):
            ret = f"    ret[{i}] = {ret_vars[i]};"
            ret_stms.append(ret)

        ret_stms.append("    return ret;")

        res.extend(ret_stms)
        res.append("}")

        res = "\n".join(res)
        return res, var_order

    def gen_verilog(self) -> str:
        res = list()
        # module def
        io_vars = [var.lexme for var in self.input_vars]
        io_vars.extend([var.lexme for var in self.output_vars])
        io_vars = ", ".join(io_vars)
        res.append(f"module {self.name.lexme}({io_vars});")
        # var declare
        for var in self.input_vars:
            res.append(f"    input {var.lexme};")
        for var in self.output_vars:
            res.append(f"    output {var.lexme};")
        for var in self.inner_vars:
            res.append(f"    wire {var.lexme};")
        # expr
        for expr in self.exprs:
            res.append(f"    assign {expr.gen_verilog()};")
        # endmodule
        res.append("endmodule")
        res = "\n".join(res)
        return res

    def __str__(self) -> str:
        res = f"Module: Name: {self.name}\n"
        res_input_vars = ""
        for token in self.input_vars:
            res_input_vars += str(token) + "\n"
        res += f"Input: \n{res_input_vars}"

        res_output_vars = ""
        for token in self.output_vars:
            res_output_vars += str(token) + "\n"
        res += f"Output: \n{res_output_vars}"
        res += "Exprs:\n"

        for expr in self.exprs:
            res += str(expr) + "\n"

        return res


def convert_skf_to_pyfunc(input: str) -> Callable:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    module.reorgnize()

    from types import FunctionType

    py_code = compile(module.gen_pycode(), "<string>", "exec")
    py_func = FunctionType(py_code.co_consts[0], globals(), module.name.lexme)

    return py_func


def convert_skf_to_pycode(input: str) -> str:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    module.reorgnize()

    return module.gen_pycode()


def convert_skf_to_cppcode(input: str) -> Tuple[str, List]:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    module.reorgnize()

    func_code, var_order = module.gen_cppcode()
    # add header
    header = [
        "#include <stdbool.h>", "#include <stdlib.h>", "#include <memory.h>"
    ]
    func_code = "\n".join(header) + "\n\n" + 'extern "C"' + "\n" + func_code
    return func_code, var_order


def convert_skf_to_forigen_func(input: str,
                                instance_name: str = "skf") -> Tuple[Callable, List]:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    module.reorgnize()

    func_code, var_order = module.gen_cppcode()

    # add header
    header = [
        "#include <stdbool.h>", "#include <stdlib.h>", "#include <memory.h>"
    ]
    func_code = "\n".join(header) + "\n\n" + 'extern "C"' + "\n" + func_code

    cpp_filename = f"run/{instance_name}.cc"
    lib_filename = f"run/lib{instance_name}.so"

    with open(cpp_filename, "w") as f:
        f.write(func_code)

    cmd = f"gcc -fPIC -shared -o {lib_filename} {cpp_filename}"
    os.system(cmd)

    # use ctypes to load the shared library
    lib = ctypes.cdll.LoadLibrary(lib_filename)
    skf_func = lib.skf_func
    skf_func.argtypes = [ctypes.POINTER(ctypes.c_bool)]
    skf_func.restype = ctypes.POINTER(ctypes.c_bool)

    return skf_func, var_order


def repair_skf_verilog(input: str) -> str:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    # module.reorgnize()
    return module.gen_verilog()


def print_debug(input: str):
    tokens = list()
    for token in Tokenzier(input):
        print(token)
        tokens.append(token)

    module = Module(tokens)
    print(module)


def get_verilog_input_order(input: str) -> List[str]:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    input_var_order = [var.lexme for var in module.input_vars]
    return input_var_order


def convert_qdimacs_to_cppcode(input: str) -> str:
    # parse to plain string
    Xvar = []
    Yvar = []
    qdimacs_list = []

    lines = input.split("\n")
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

        clause = line.strip(" ").strip("\n").strip(" ").split(" ")[:-1]
        if len(clause) > 0:
            clause = list(map(int, list(clause)))
            qdimacs_list.append(clause)
    Xvar = list(map(int, Xvar))
    Yvar = list(map(int, Yvar))
    Tvar = list()
    # check var in clause
    for clause in qdimacs_list:
        for var in clause:
            if abs(var) not in Xvar and abs(var) not in Yvar and abs(var) not in Tvar:
                Tvar.append(abs(var))

    # gen cpp code
    def special_sort(vars: list) -> list:
        for i in range(1, len(vars)):
            for j in range(0, len(vars) - i):
                var_1 = int(vars[j][1:])
                var_2 = int(vars[j + 1][1:])
                if var_1 > var_2:
                    vars[j], vars[j + 1] = vars[j + 1], vars[j]
        return vars

    res = list()
    vars_list = ["x" + str(xvar) for xvar in Xvar]
    vars_list.extend(["y" + str(yvar) for yvar in Yvar])
    vars_list.extend(["t" + str(tvar) for tvar in Tvar])
    vars_list = special_sort(vars_list)

    # func def
    res.append("bool F_func(bool* args)")
    res.append("{")
    for i in range(len(vars_list)):
        res.append(f"    bool {vars_list[i]} = args[{i}];")
    # expr
    exprs_def = list()
    for clause in qdimacs_list:
        clause_expr = []
        for var in clause:
            is_xvar = False
            is_yvar = False
            is_tvar = False
            if abs(var) in Xvar:
                is_xvar = True

            if abs(var) in Yvar:
                is_yvar = True

            if abs(var) in Tvar:
                is_tvar = True

            if var < 0:
                if is_xvar:
                    literal = "!" + "x" + str(abs(var))
                    clause_expr.append(literal)
                elif is_yvar:
                    literal = "!" + "y" + str(abs(var))
                    clause_expr.append(literal)
                elif is_tvar:
                    literal = "!" + "t" + str(abs(var))
                    clause_expr.append(literal)
            else:
                if is_xvar:
                    literal = "x" + str(var)
                    clause_expr.append(literal)
                elif is_yvar:
                    literal = "y" + str(var)
                    clause_expr.append(literal)
                elif is_tvar:
                    literal = "t" + str(var)
                    clause_expr.append(literal)
        clause_expr = " || ".join(clause_expr)
        clause_expr = "( " + clause_expr + " )"
        exprs_def.append(clause_expr)
    exprs_def = " && ".join(exprs_def)
    exprs_def = "    bool out = " + exprs_def + ";"
    res.append(exprs_def)
    # return
    ret = "    return out;"
    res.append(ret)
    res.append("}")
    func_code = "\n".join(res)

    header = [
        "#include <stdbool.h>", "#include <stdlib.h>", "#include <memory.h>"
    ]
    func_code = "\n".join(header) + "\n\n" + 'extern "C"' + "\n" + func_code

    return func_code


def convert_qdimacs_to_forigen_func(input: str,
                                    instance_name: str = "F_qdimacs"):
    cpp_code = convert_qdimacs_to_cppcode(input)
    instance_name = "F_" + to_valid_filename(instance_name)

    cpp_filename = f"run/{instance_name}.cc"
    lib_filename = f"run/lib{instance_name}.so"

    with open(cpp_filename, "w") as f:
        f.write(cpp_code)

    cmd = f"gcc -fPIC -shared -o {lib_filename} {cpp_filename}"
    os.system(cmd)

    # use ctypes to load the shared library
    lib = ctypes.cdll.LoadLibrary(lib_filename)
    F_func = lib.F_func
    F_func.argtypes = [ctypes.POINTER(ctypes.c_bool)]
    F_func.restype = ctypes.c_bool

    return F_func
