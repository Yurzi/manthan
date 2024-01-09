#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:39:16 2020

@author: fs
"""


def miniSAT_literals(literals):
    return [2 * abs(lit) + (lit < 0) for lit in literals]


def miniSAT_clauses(clauses):
    return [miniSAT_literals(c) for c in clauses]


def maxVarIndex(clause_list):
    return max([abs(l) for c in clause_list for l in c], default=0)


def clausalEncodingAND(definition):
    defining_literals, defined_literal = definition
    negative_clauses = [[l, -defined_literal] for l in defining_literals]
    positive_clause = [-l for l in defining_literals] + [defined_literal]
    return negative_clauses + [positive_clause]


def renameLiteral(l, renaming):
    return renaming.get(l, l) if l > 0 else -renaming.get(abs(l), abs(l))


def renameClause(clause, renaming):
    return [renameLiteral(l, renaming) for l in clause]


def renameFormula(clauses, renaming):
    return [renameClause(clause, renaming) for clause in clauses]


def createRenaming(clauses, shared_variables, auxiliary_start=None):
    variables = {abs(l) for clause in clauses for l in clause}
    non_shared = variables.difference(shared_variables)
    if auxiliary_start is None:
        auxiliary_start = max(variables, default=0) + 1
    renaming_range = range(auxiliary_start, auxiliary_start + len(non_shared))
    renaming = dict(zip(non_shared, renaming_range))
    renamed_clauses = renameFormula(clauses, renaming)
    return renamed_clauses, renaming, auxiliary_start + len(non_shared) - 1


def negate(clauses, auxiliary_start=None):
    if auxiliary_start is None:
        auxiliary_start = maxVarIndex(clauses) + 1
    auxiliary_range = range(auxiliary_start, auxiliary_start + len(clauses) + 1)
    small_clauses = [
        [-l, auxiliary_range[i]] for i in range(len(clauses)) for l in clauses[i]
    ]
    big_clause = [-v for v in auxiliary_range]
    return small_clauses + [big_clause]


def equality(lit1, lit2, switch):
    return [[-switch, lit1, -lit2], [-switch, -lit1, lit2]]


class XGBoostTreeNode:
    def __init__(self):
        self.nodeid: int = -1
        self.depth: int = -1

        self.parent = None
        self.left = None
        self.right = None

        self.split = None
        self.split_condition: float = 0

        self.leaf = 0

        self.is_leaf: bool = False

    def from_dict(self, input):
        # node id
        self.nodeid = input["nodeid"]

        if input.get("depth") is not None:
            self.depth = input["depth"]
        else:
            if self.parent is not None:
                self.depth = self.parent.depth + 1
            else:
                self.depth = 0
        if input.get("split") is not None:
            self.split = input["split"]
        else:
            self.split = None

        if input.get("split_condition") is not None:
            self.split_condition = input["split_condition"]
        else:
            self.split_condition = 0

        if input.get("leaf") is not None:
            self.leaf = input["leaf"]
        else:
            self.leaf = 0

        if input.get("children") is not None:
            self.is_leaf = False
            for child in input["children"]:
                if child["nodeid"] == input["yes"]:
                    self.left = XGBoostTreeNode()
                    self.left.parent = self
                    self.left.from_dict(child)
                elif child["nodeid"] == input["no"]:
                    self.right = XGBoostTreeNode()
                    self.right.parent = self
                    self.right.from_dict(child)
                else:
                    raise ValueError("Invalid child node id")
        else:
            self.is_leaf = True

    def __dict__(self):
        res = {}
        res["nodeid"] = self.nodeid
        res["depth"] = self.depth
        res["split"] = self.split
        res["split_condition"] = self.split_condition
        res["leaf"] = self.leaf
        if not self.is_leaf:
            res["children"] = []
            res["yes"] = self.left.nodeid
            res["no"] = self.right.nodeid

            res["children"].append(self.left.__dict__())
            res["children"].append(self.right.__dict__())

        return res

    def treepath(self, dependson, Xvar, Yvar, index, size, args):
        if self.is_leaf:
            if self.leaf >= 0:
                return (["1"], dependson)
            else:
                return (["val=0"], dependson)

        assert self.left is not None
        left_subtree, dependson = self.left.treepath(
            dependson, Xvar, Yvar, index, size, args
        )
        assert self.right is not None
        right_subtree, dependson = self.right.treepath(
            dependson, Xvar, Yvar, index, size, args
        )

        # conjunction of all the literal in a path where leaf node has label 1
        # Dependson is list of Y variables on which candidate SKF of y_i depends
        list_left = []
        for leaf in left_subtree:
            if leaf != "val=0":
                if int(self.split) in Yvar:
                    dependson.append(int(self.split))
                    # the left part
                    list_left.append("~w" + str(self.split) + " & " + leaf)
                else:
                    list_left.append("~i" + str(self.split) + " & " + leaf)

        list_right = []
        for leaf in right_subtree:
            if leaf != "val=0":
                if int(self.split) in Yvar:
                    dependson.append(int(self.split))
                    list_right.append("w" + str(self.split) + " & " + leaf)
                else:
                    list_right.append("i" + str(self.split) + " & " + leaf)

        dependson = list(set(dependson))

        return (list_left + list_right, dependson)


class UniqueCalcMap:

    class CalcNode:
        def __init__(self, id, is_neg=False) -> None:
            self.id = id
            self.input = []
            self.output = []
            self.value = None
            self.is_neg = is_neg

        def calc(self):
            res = self.value
            if len(self.input) >= 1:
                if len(self.input) == 1:
                    res = self.input[0].value
                else:
                    res = [node.value for node in self.input]
                    res = all(res)
                    
            if self.is_neg:
                res = not res

            self.value = res

        def __dict__(self):
            res = {}
            res["id"] = self.id
            res["input"] = [node.id for node in self.input]
            res["output"] = [node.id for node in self.output]
            res["value"] = self.value
            res["is_neg"] = self.is_neg
            return res
        
        def __str__(self):
            return str(self.__dict__())
            
    def __init__(self, Xvar, Yvar, UniqueVar, TempVar, UniqueRawDef):
        self.Xvar = Xvar
        self.Yvar = Yvar
        self.UniqueVar = UniqueVar
        self.TempVar = TempVar
        self.constant = []

        self.CalcMap = {}

        self.constant.append(self.CalcNode("cons-0"))
        self.constant[0].value = False
        self.constant.append(self.CalcNode("cons-1"))
        self.constant[1].value = True

        for var in Xvar:
            node = self.CalcNode(var)
            self.CalcMap[var] = node
            node_neg = self.CalcNode(-var, True)
            self.CalcMap[-var] = node_neg

        for var in TempVar:
            node = self.CalcNode(var)
            self.CalcMap[var] = node
            node_neg = self.CalcNode(-var, True)
            self.CalcMap[-var] = node_neg

        for var in Yvar:
            node = self.CalcNode(var)
            self.CalcMap[var] = node
            node_neg = self.CalcNode(-var, True)
            self.CalcMap[-var] = node_neg

        for out, inputs in UniqueRawDef.items():
            if not isinstance(inputs, list):
                self.CalcMap[out] = self.constant[inputs]
                self.CalcMap[-out] = self.constant[1 - inputs]
                continue

            for input in inputs:
                self.CalcMap[out].input.append(self.CalcMap[input])
                self.CalcMap[-out].input.append(self.CalcMap[input])
                self.CalcMap[input].output.append(self.CalcMap[out])
                self.CalcMap[input].output.append(self.CalcMap[-out])

        # split in layers
        self.layers = []
        vectors = {}

        is_visited = {}
        for i, node in self.CalcMap.items():
            vectors[i] = len(node.input)
            is_visited[i] = False

        # decrease vectors due to constant
        for node in self.constant[0].output:
            vectors[node] -= 1
        for node in self.constant[1].output:
            vectors[node] -= 1

        # init
        stack = []
        stack_next = []
        for idx, value in vectors.items():
            if value == 0:
                stack_next.append(idx)

        while True:
            if len(stack) == 0:
                if len(stack_next) == 0:
                    break
                else:
                    self.layers.append(stack_next.copy())
                    stack = stack_next
                    stack_next = []
                    continue

            node_idx = stack.pop()
            node = self.CalcMap[node_idx]
            if is_visited[node_idx] is True:
                continue

            is_visited[node_idx] = True
            for output in node.output:
                vectors[output.id] -= 1
                if vectors[output.id] == 0:
                    stack_next.append(output.id)

    def __call__(self, samples):
        for sample in samples:
            for xvar in self.Xvar:
                self.CalcMap[xvar].value = bool(sample[xvar - 1])
                self.CalcMap[-xvar].value = bool(sample[xvar - 1])
            for yvar in self.Yvar:
                self.CalcMap[yvar].value = bool(sample[yvar - 1])
                self.CalcMap[-yvar].value = bool(sample[yvar - 1])
            for layer in self.layers:
                for node_idx in layer:
                    self.CalcMap[node_idx].calc()

            for yvar in self.Yvar:
                sample[yvar - 1] = int(self.CalcMap[yvar].value)

    def print_map(self):
        print("-------Layers--------=--")
        print(self.layers)
        print("-------Inner MAP--------")
        for i, node in self.CalcMap.items():
            print(node)
