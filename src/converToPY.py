from collections import defaultdict
from enum import Enum
from typing import Self


class Graph:
    def __init__(self, vertexs: list) -> None:
        self._graph: defaultdict = defaultdict(list)
        self._vertexs = set(vertexs)

        # init
        for v in self._vertexs:
            self._graph[v] = list()

    def add_edge(self, u, v) -> None:
        # check u, v
        if u not in self._vertexs:
            raise ValueError(f"Vertex {u} is not in graph")
        if v not in self._vertexs:
            raise ValueError(f"Vertex {v} is not in graph")

        self._graph[u].append(v)

    def topologic_sort(self) -> list:
        income_count: defaultdict = defaultdict(lambda: 0)
        visted: defaultdict = defaultdict(lambda: False)
        result: list = list()

        # init income_count
        for v in self._vertexs:
            for edge in self._graph[v]:
                income_count[edge] += 1

        pending_list = list()
        for v in self._vertexs:
            if income_count[v] == 0:
                pending_list.append(v)

        while len(pending_list) > 0:
            v = pending_list.pop(0)
            if visted[v]:
                continue

            visted[v] = True

            for edge in self._graph[v]:
                if income_count[edge] > 0:
                    income_count[edge] -= 1

                if income_count[edge] == 0:
                    pending_list.append(edge)

            result.append(v)

        for v in self._vertexs:
            if visted[v]:
                continue
            raise ValueError(f"Graph has loop, start vertex is {v}")

        return result


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
            if self._index + 1 >= len(self._chars):
                return self.EOF_CHAR

            return self._chars[self._index]

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
    def is_lit_contine(c: str) -> bool:
        return c.isdigit() or c == "_"

    def lit(self) -> None:
        now_char = self.cursor.bump()  # eat ' or a digit or b or o or h
        if now_char == "'":
            self.cursor.bump()  # eat b or o or h
            self.cursor.eat_while(self.is_lit_contine)
            return
        if not now_char.isdigit():
            self.cursor.eat_while(self.is_lit_contine)
            return
        self.cursor.eat_while(self.is_lit_contine)
        self.cursor.bump()  # eat '
        self.cursor.bump()  # eat b or o or h
        self.cursor.eat_while(self.is_lit_contine)

    @staticmethod
    def is_keyword(lexme: str) -> bool:
        if lexme in ["module", "input", "output", "wire", "assign", "endmodule"]:
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
                literal = token.lexme.split("'")[-1]
                literal = literal[1:]
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

    def gen_verilog(self) -> str:
        res = []
        for token in self.expr:
            if token.kind is Token.Kind.And:
                res.append("&")
                continue
            if token.kind is Token.Kind.Or:
                res.append("|")
                continue
            if token.kind is Token.Kind.Not:
                res.append("~")
                continue
            if token.kind is Token.Kind.Xor:
                res.append("^")
                continue
            if token.kind is Token.Kind.Literal:
                if int(token.lexme) == 1:
                    res.append("1")
                else:
                    if int(token.lexme) == 0:
                        res.append("1")
                continue
            res.append(token.lexme)
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

        vertex.sort()

        graph = Graph(vertex)
        for expr in self.exprs:
            for token in expr.income:
                if token.kind is Token.Kind.Literal:
                    continue

                graph.add_edge(token.lexme, expr.outcome.lexme)

        sorted_vars = graph.topologic_sort()
        sorted_exprs = list()
        for expr_out in sorted_vars:
            for expr in self.exprs:
                if expr.outcome.lexme == expr_out:
                    sorted_exprs.append(expr)
        assert len(sorted_exprs) == len(self.exprs), "some experssion is missing"
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


def repair_skf_verilog(input: str) -> str:
    tokens = list()
    for token in Tokenzier(input):
        tokens.append(token)

    module = Module(tokens)
    module.reorgnize()
    return module.gen_verilog()
