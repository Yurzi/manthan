import numpy as np
from xgboost import DMatrix

import sympy


class CustomL1Loss:
    def __init__(self) -> None:
        pass

    def __call__(self, predt: np.ndarray, dtrain: DMatrix):
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
        predt = predt.reshape(-1, 1)
        label = dtrain.get_label().reshape(-1, 1)
        grad = 2 * (predt - label)
        hess = np.repeat(2, label.shape[0])
        return grad, hess

    def metrics(self, predt, dtrain):
        predt = predt.reshape(-1, 1)
        y = (predt - dtrain.get_label().reshape(-1, 1)) ** 2
        return "L1-Loss", np.mean(y)


class CustomL2Loss:
    def __init__(self,
                 qdimacs: str,
                 featname,
                 labelname,
                 PosUnate,
                 NegUnate,
                 full_samples) -> None:
        self.qdimacs = qdimacs
        self.clusters = []
        self.vars = {}
        self.expr = None

        self.full_samples = full_samples
        self.featname = featname
        self.labelname = labelname
        self.pos_vars = PosUnate
        self.neg_vars = NegUnate

        self.grad_compiled = None
        self.hess_compiled = None
        self.expr_compiled = None

        self.input_vars = None
        self.pred_y = None

        self.parse_qdimacs()

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

    def compile(self):
        # self.expr_compiled = sympy.lambdify(self.vars.values(), self.expr, "numpy")
        grad_res = 0
        hess_res = 0

        for yvar in self.labelname:
            grad = self.expr.diff(self.vars[str(yvar)])
            grad_res += grad
            hess_res += grad.diff(self.vars[str(yvar)])

        self.grad_compiled = sympy.lambdify(self.vars.values(), grad_res, "numpy")
        self.hess_compiled = sympy.lambdify(self.vars.values(), grad_res, "numpy")

    def reshape_data(self, predt: np.ndarray, dtrain: DMatrix):
        predt = predt.reshape(-1, len(self.labelname))
        predt_res = {}

        features_data = dtrain.get_data().toarray()
        for idx, xvar in enumerate(self.featname):
            predt_res[self.vars[str(xvar)].name] = features_data[:, idx]

        for idx, yvar in enumerate(self.labelname):
            predt_res[self.vars[str(yvar)].name] = predt[:, idx]

        for yvar in self.pos_vars:
            # predt_res[self.vars[str(yvar)].name] = np.ones(features_data.shape[0])
            predt_res[self.vars[str(yvar)].name] = self.full_samples[:, yvar - 1]

        for yvar in self.neg_vars:
            # predt_res[self.vars[str(yvar)].name] = np.zeros(features_data.shape[0])
            predt_res[self.vars[str(yvar)].name] = self.full_samples[:, yvar - 1]

        left_var = (set([int(key) for key in self.vars.keys()])
                    - set(self.featname)
                    - set(self.labelname)
                    - set(self.pos_vars)
                    - set(self.neg_vars))

        for var in left_var:
            predt_res[self.vars[str(var)].name] = self.full_samples[:, var - 1]

        # samples = self.merge_from_dict(predt_res)

        # predt_res = self.split_to_dict(samples)
        # print(predt_res)

        return predt_res

    # 从字典合并数组
    def merge_from_dict(self, data: dict):
        res = []
        num_vars = len(self.vars.keys())
        for i in range(1, num_vars + 1):
            res.append(data[self.vars[str(i)].name])

        res = np.array(res).T
        return res

    # 分解为字典
    def split_to_dict(self, data: np.ndarray):
        res = {}
        num_vars = len(self.vars.keys())
        for i in range(1, num_vars + 1):
            res[self.vars[str(i)].name] = data[:, i - 1]

        return res

    def __call__(self, predt: np.ndarray, dtrain: DMatrix):
        input = self.reshape_data(predt, dtrain)

        grad = -self.grad_compiled(**input) / predt.shape[0]
        hess = -self.hess_compiled(**input) / predt.shape[0]
        return grad, hess

    def metrics(self, predt, dtrain):
        input = self.reshape_data(predt, dtrain)
        y = 1 - self.expr_compiled(**input) / predt.shape[0]
        return "L2-Loss", np.mean(y)
