import ctypes
import os
from enum import Enum
from typing import Self

import networkx as nx


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
        self.kind: Self.Kind = kind
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

        def eat_while(self, predicate: callable) -> None:
            while predicate(self.first) and not self.is_eof:
                self.bump()

    def __init__(self, input: str) -> None:
        self.raw_str: str = input
        self.cursor: self.Cursor = self.Cursor(input)

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
            return Token(Token.Kind.Eof, 0)

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

    def __init__(self, lineno: int, tokens: list[Token]) -> None:
        self.kind: self.Kind = None
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
    def from_tokens(tokens: list[Token]) -> list[Self]:
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
        self.outcome: Token = None
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
        self.name: Token | None = None
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

    def gen_pycode(self) -> str:

        def special_sort(vars: list) -> list:
            for i in range(1, len(vars)):
                for j in range(0, len(vars) - i):
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

    def gen_cppcode(self):

        def special_sort(vars: list) -> list:
            for i in range(1, len(vars)):
                for j in range(0, len(vars) - i):
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

        res.append("bool* skf_func(bool* args)")
        res.append("{")

        for i in range(len(var_list)):
            res.append("    bool " + var_list[i] + " = args[" + str(i) + "];")
        # allocate memory for return
        ret_vars = [var.lexme for var in self.output_vars]
        ret_vars.extend([var.lexme for var in self.input_vars])
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
        return res

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


def convert_skf_to_pyfunc(input: str) -> callable:
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


def convert_skf_to_cppcode(input: str) -> str:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    module.reorgnize()

    func_code = module.gen_cppcode()
    # add header
    header = [
        "#include <stdbool.h>", "#include <stdlib.h>", "#include <memory.h>"
    ]
    func_code = "\n".join(header) + "\n\n" + 'extern "C"' + "\n" + func_code
    return func_code


def convert_skf_to_forigen_func(input: str, instance_name: str = "skf") -> callable:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    module.reorgnize()

    func_code = module.gen_cppcode()

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

    return skf_func


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
