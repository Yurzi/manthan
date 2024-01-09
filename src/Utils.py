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