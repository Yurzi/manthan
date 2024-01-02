#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Priyanka Golia, Subhajit Roy, and Kuldeep Meel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import collections
import json
from typing import Any

import networkx as nx
import numpy as np
import pydotplus
import xgboost as xgb
from sklearn import tree
from xgboost import DMatrix
import sympy


def treepaths(
    root,
    is_leaves,
    children_left,
    children_right,
    data_feature_names,
    feature,
    values,
    dependson,
    leave_label,
    Xvar,
    Yvar,
    index,
    size,
    args,
):
    if is_leaves[root]:
        if not args.multiclass:
            temp = values[root]
            temp = temp.ravel()
            if len(temp) == 1:
                if leave_label[0] == 1:
                    return (["1"], dependson)
                else:
                    return (["val=0"], dependson)

            if temp[1] < temp[0]:
                return (["val=0"], dependson)
            else:
                return (["1"], dependson)
        else:
            node_label = leave_label[root]
            bool_res = format(node_label, "0" + str(size) + "b")
            if int(bool_res[index]):
                return (["1"], dependson)
            else:
                return (["val=0"], dependson)

    left_subtree, dependson = treepaths(
        children_left[root],
        is_leaves,
        children_left,
        children_right,
        data_feature_names,
        feature,
        values,
        dependson,
        leave_label,
        Xvar,
        Yvar,
        index,
        size,
        args,
    )
    right_subtree, dependson = treepaths(
        children_right[root],
        is_leaves,
        children_left,
        children_right,
        data_feature_names,
        feature,
        values,
        dependson,
        leave_label,
        Xvar,
        Yvar,
        index,
        size,
        args,
    )

    # conjunction of all the literal in a path where leaf node has label 1
    # Dependson is list of Y variables on which candidate SKF of y_i depends
    list_left = []
    for leaf in left_subtree:
        if leaf != "val=0":
            if data_feature_names[feature[root]] in Yvar:
                dependson.append(data_feature_names[feature[root]])
                # the left part
                list_left.append(
                    "~w" + str(data_feature_names[feature[root]]) + " & " + leaf
                )
            else:
                list_left.append(
                    "~i" + str(data_feature_names[feature[root]]) + " & " + leaf
                )

    list_right = []
    for leaf in right_subtree:
        if leaf != "val=0":
            if data_feature_names[feature[root]] in Yvar:
                dependson.append(data_feature_names[feature[root]])
                list_left.append(
                    "w" + str(data_feature_names[feature[root]]) + " & " + leaf
                )
            else:
                list_left.append(
                    "i" + str(data_feature_names[feature[root]]) + " & " + leaf
                )
    dependson = list(set(dependson))

    return (list_left + list_right, dependson)


def createDecisionTree(featname, featuredata, labeldata, yvar, args, Xvar, Yvar):
    clf = tree.DecisionTreeClassifier(
        criterion="gini", min_impurity_decrease=args.gini, random_state=args.seed
    )
    clf = clf.fit(featuredata, labeldata)
    if args.showtrees:
        dot_data = tree.export_graphviz(
            clf, feature_names=featname, out_file=None, filled=True, rounded=True
        )
        graph = pydotplus.graph_from_dot_data(dot_data)
        colors = ("turquoise", "orange")
        edges = collections.defaultdict(list)
        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))
        for edge in edges:
            edges[edge].sort()
            for i in range(2):
                dest = graph.get_node(str(edges[edge][i]))[0]
                dest.set_fillcolor(colors[i])
        graph.write_png(str(yvar) + ".png")
    values = clf.tree_.value
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    # threshold = clf.tree_.threshold
    leaves = children_left == -1
    leaves = np.arange(0, n_nodes)[leaves]
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    leave_label = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
            leave_label[node_id] = clf.classes_[np.argmax(clf.tree_.value[node_id])]

    D_dict = {}
    psi_dict = {}

    for i in range(len(yvar)):
        D = []
        paths, D = treepaths(
            0,
            is_leaves,
            children_left,
            children_right,
            featname,
            feature,
            values,
            D,
            leave_label,
            Xvar,
            Yvar,
            i,
            len(yvar),
            args,
        )

        psi_i = ""

        if is_leaves[0]:
            if "val=0" in paths:
                paths = ["0"]
            else:
                paths = ["1"]

        if len(paths) == 0:
            paths.append("0")
            D = []

        for path in paths:
            psi_i += "( " + path + " ) | "

        D_dict[yvar[i]] = D
        psi_dict[yvar[i]] = psi_i.strip("| ")

    return psi_dict, D_dict


def binary_to_int(lst):
    lst = np.array(lst)
    # filling the begining with zeros to form bytes
    diff = 8 - lst.shape[1] % 8
    if diff > 0 and diff != 8:
        lst = np.c_[np.zeros((lst.shape[0], diff), int), lst]
    label = np.packbits(lst, axis=1)

    return label


def createCluster(args, Yvar, SkolemKnown, ng):
    disjointSet = []
    clusterY = []

    for var in Yvar:
        if var in SkolemKnown:
            continue

        if (args.multiclass) and not (args.henkin):
            if var in list(ng.nodes):
                Yset = []
                hoppingDistance = args.hop

                while hoppingDistance > 0:
                    hop_neighbour = list(
                        nx.single_source_shortest_path_length(
                            ng, var, cutoff=hoppingDistance
                        )
                    )

                    if len(hop_neighbour) < args.clustersize:
                        break
                    else:
                        hop_neighbour = []
                    hoppingDistance -= 1

                if len(hop_neighbour) == 0:
                    hop_neighbour = [var]

                for var2 in hop_neighbour:
                    ng.remove_node(var2)
                    Yset.append(var2)
                    clusterY.append(var2)  # list of all variables cluster so far.

                disjointSet.append(Yset)  # list of lists
            else:
                if var not in clusterY:
                    disjointSet.append([var])
        else:
            disjointSet.append([var])

    return disjointSet


def learnCandidate(
    Xvar, Yvar, UniqueVars, PosUnate, NegUnate, samples, dg, ng, args, HenkinDep={}
):
    candidateSkf = (
        {}
    )  # represents y_i and its corresponding learned candidate via decision tree.

    SkolemKnown = PosUnate + NegUnate + UniqueVars

    for var in SkolemKnown:
        if (args.multiclass) and (var in list(ng.nodes)):
            ng.remove_node(var)

        for var in PosUnate:
            candidateSkf[var] = " 1 "

        for var in NegUnate:
            candidateSkf[var] = " 0 "

    disjointSet = createCluster(args, Yvar, SkolemKnown, ng)

    for Yset in disjointSet:
        dependent = []
        for yvar in Yset:
            if not args.henkin:
                depends_on_yvar = list(nx.ancestors(dg, yvar))
                depends_on_yvar.append(yvar)
                dependent = dependent + depends_on_yvar
            else:
                yvar_depends_on = list(nx.descendants(dg, yvar))
                if yvar in list(yvar_depends_on):
                    yvar_depends_on.remove(yvar)

        if not args.henkin:
            Yfeatname = list(set(Yvar) - set(dependent))
            featname = Xvar.copy()
            samples_X = samples[:, (np.array(Xvar) - 1)]
        else:
            Yfeatname = yvar_depends_on
            featname = HenkinDep[Yset[0]]
            samples_X = samples[:, (np.array(HenkinDep[Yset[0]]) - 1)]

        if len(Yfeatname) > 0:
            featname += Yfeatname
            Samples_Y = samples[:, (np.array(Yfeatname) - 1)]
            featuredata = np.concatenate((samples_X, Samples_Y), axis=1)

        else:
            featuredata = samples_X

        label = samples[:, (np.array(Yset) - 1)]
        labeldata = binary_to_int(label)

        assert len(featname) == len(featuredata[0])
        assert len(Yset) == len(labeldata[0])

        functions, D_set = createXGBDecisionTree(featname,
                                                 featuredata,
                                                 labeldata,
                                                 Yset,
                                                 args,
                                                 Xvar,
                                                 Yvar,
                                                 PosUnate,
                                                 NegUnate,
                                                 samples)

        # functions, D_set = createDecisionTree(
        #     featname, featuredata, labeldata, Yset, args, Xvar, Yvar
        # )

        for var in functions.keys():
            assert var not in UniqueVars
            assert var not in PosUnate
            assert var not in NegUnate
            candidateSkf[var] = functions[var]
            D = list(set(D_set[var]) - set(Xvar))
            for jvar in D:
                dg.add_edge(var, jvar)

    if args.verbose:
        print(" c generated candidate functions for all variables.")

    if args.verbose == 2:
        print(" c candidate functions are", candidateSkf)

    return candidateSkf, dg


class TreeNode:
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
                    self.left = TreeNode()
                    self.left.parent = self
                    self.left.from_dict(child)
                elif child["nodeid"] == input["no"]:
                    self.right = TreeNode()
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


def CustomL1Loss(predt: np.ndarray, dtrain: DMatrix):
    """Custom L1 loss function for XGBoost.

    Parameters
    ----------
    predt : np.ndarray
        The predicted values.
    dtrain : DMatrix
        The training data.

    Returns
    -------
    grad : np.ndarray
        The first order gradients.
    hess : np.ndarray
        The second order gradients.
    """
    label = dtrain.get_label()
    grad = 2 * (predt - label)
    hess = np.repeat(2, label.shape[0])
    return grad, hess


class CustomL2Loss():
    def __init__(self, qdimacs: str, samples, PosUnate, NegUnate) -> None:
        self.qdimacs = qdimacs
        self.clusters = []
        self.vars = {}
        self.expr = None
        self.samples = samples

        self.grad = None
        self.hess = None

        self.grad_compiled = None
        self.hess_compiled = None

        self.input_vars = None
        self.pred_y = None

        self.parse_qdimacs()
        self.set_constant(PosUnate, NegUnate)

    def parse_qdimacs(self) -> None:
        # parse to plain string
        Xvar = []
        Yvar = []
        qdimacs_list = []

        lines = self.qdimacs.split("\n")
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
                continue

            clause = line.strip(" ").strip("\n").strip(" ").split(" ")[:-1]

            if len(clause) > 0:
                clause = list(map(int, list(clause)))
                qdimacs_list.append(clause)

        # convert to sympy expression
        for xvar in Xvar:
            self.vars[xvar] = sympy.Symbol('x' + xvar)
        for yvar in Yvar:
            self.vars[yvar] = sympy.Symbol('y' + yvar)

        for clause in qdimacs_list:
            tmp_exprs = []
            for lit in clause:
                if lit < 0:
                    tmp_expr = 1 - self.vars[str(-lit)]
                    tmp_exprs.append(tmp_expr)
                else:
                    tmp_expr = self.vars[str(lit)]
                    tmp_exprs.append(tmp_expr)

            if len(tmp_exprs) < 1:
                continue
            if len(tmp_exprs) == 1:
                self.clusters.append(tmp_exprs[0])
            else:
                expr = tmp_exprs[0]
                for tmp_expr in tmp_exprs[1:]:
                    expr = expr + tmp_expr
                self.clusters.append(expr)

        if len(self.clusters) < 1:
            raise ValueError("No cluster found")
        if len(self.clusters) == 1:
            self.expr = self.clusters[0]
        else:
            expr = self.clusters[0]
            for cluster in self.clusters[1:]:
                expr = expr * cluster
            self.expr = expr

    def set_constant(self, PosUnate, NegUnate) -> None:
        for var in PosUnate:
            self.expr = self.expr.subs(self.vars[str(var)], 1)
        for var in NegUnate:
            self.expr = self.expr.subs(self.vars[str(var)], 0)

    def set_used_var(self, relative_var, indep_var: int):
        self.input_vars = {}
        for var in relative_var:
            self.input_vars[str(var)] = self.vars[str(var)].name
        self.pred_y = self.vars[str(indep_var)].name

        for var in self.vars.keys():
            if int(var) not in relative_var and int(var) != indep_var:
                # 将无法用到的变量设置为 0.5
                self.expr = self.expr.subs(self.vars[var], 0.5)

        self.expr = 1 - self.expr

        self.grad = sympy.diff(self.expr, self.vars[str(indep_var)])
        self.hess = sympy.diff(self.grad, self.vars[str(indep_var)])

        used_vars = [self.vars[str(i)] for i in relative_var]
        used_vars += [self.vars[str(indep_var)]]
        self.grad_compiled = sympy.lambdify(used_vars, self.grad, "numpy")
        self.hess_compiled = sympy.lambdify(used_vars, self.hess, "numpy")


    def __call__(self, predt: np.ndarray, dtrain: DMatrix):
        input = {}
        input[self.pred_y] = predt
        index = 0
        for var in dtrain.feature_names:
            input[self.input_vars[var]] = np.array(dtrain.get_data().toarray()[:, index])
            index += 1
            
        grad = self.grad_compiled(**input)
        hess = self.hess_compiled(**input)

        return grad, hess
    

class CustomMixLoss():
    def __init__(self, alpha, beta, qdimacs: str, samples, PosUnate, NegUnate):
        self.alpha = alpha
        self.beta = beta

        self.l1 = CustomL1Loss
        self.l2 = CustomL2Loss(qdimacs, samples, PosUnate, NegUnate)

    def set_used_var(self, relative_var, indep_var: int):
        self.l2.set_used_var(relative_var, indep_var)

    def __call__(self, predt: np.ndarray, dtrain: DMatrix):
        grad1, hess1 = self.l1(predt, dtrain)
        grad2, hess2 = self.l2(predt, dtrain)

        grad = self.alpha * grad1 + self.beta * grad2
        hess = self.alpha * hess1 + self.beta * hess2

        return grad, hess
    

def createXGBDecisionTree(featname,
                          featuredata,
                          labeldata, yvar,
                          args,
                          Xvar, 
                          Yvar,
                          PosUnate,
                          NegUnate,
                          samples):
    xgb_feature_names = [str(featname[i]) for i in range(len(featname))]
    xgb_params = {
        "objective": "binary:logistic",
    }
    xgb_dtrain = DMatrix(data=featuredata,
                         label=labeldata,
                         feature_names=xgb_feature_names)
    # xgb_dtrain_gpu = QuantileDMatrix(data=featuredata,
    #                                  label=labeldata,
    #                                  feature_names=xgb_feature_names)

    # custom_l2_loss = CustomL2Loss(args.qdimacsstr, samples, PosUnate, NegUnate)
    # custom_l2_loss.set_used_var(featname, yvar[0])

    custom_mix_loss = CustomMixLoss(0.5, 0.5, args.qdimacsstr, samples, PosUnate, NegUnate)
    custom_mix_loss.set_used_var(featname, yvar[0])

    xgb_clf = xgb.train(params=xgb_params,
                        dtrain=xgb_dtrain,
                        num_boost_round=1,
                        obj=custom_mix_loss)

    # dump tree json
    tree_json = xgb_clf.get_dump(with_stats=True, dump_format="json")
    if args.showtrees:
        xgb_clf.dump_model("xgb_model.json", with_stats=True, dump_format="json")

    assert len(tree_json) == 1
    tree_dict = json.loads(tree_json[0])
    tree = TreeNode()
    tree.from_dict(tree_dict)

    D_dict = {}
    psi_dict = {}

    for i in range(len(yvar)):
        D = []
        paths, D = tree.treepath(D, Xvar, Yvar, i, len(yvar), args)

        psi_i = ""
        if tree.is_leaf:
            if "val=0" in paths:
                paths = ["0"]
            else:
                paths = ["1"]

        if len(paths) == 0:
            paths.append("0")
            D = []

        for path in paths:
            psi_i += "( " + path + " ) | "

        D_dict[yvar[i]] = D
        psi_dict[yvar[i]] = psi_i.strip("| ")

    return psi_dict, D_dict
