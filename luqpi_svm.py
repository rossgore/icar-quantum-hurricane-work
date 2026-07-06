"""
SVM+ (Vapnik & Vashist, 2009) — the classical learning algorithm used to
implement Learning Under Quantum Privileged Information (LUQPI).

Quantum-derived features (here: the Phase 4 classical-shadow PCA components)
are passed in only as "privileged information" x* during fit(): they shape
the slack variables of a soft-margin SVM via a second RBF kernel, guiding
the training of a more informed decision boundary. predict() never touches
x* — the trained model needs only the ordinary classical features x, so no
quantum computation is required at deployment time.

Dual QP (variables z = [alpha; beta] in R^2n), from the SVM+ derivation:

  minimize   (1/2) z^T P z + q^T z
  subject to y^T alpha = 0,  sum(alpha + beta) = n*C,  alpha >= 0,  beta >= 0

  P = [[ Y K Y + Ks/C*,  Ks/C* ],
       [ Ks/C*,          Ks/C* ]]
  q = [ -1 - (C/C*) Ks@1 ;  -(C/C*) Ks@1 ]

where K is the RBF kernel on x, Ks is the RBF kernel on x*, and Y = diag(y).
Solved with cvxopt; bias terms b (decision) and b* (correcting function)
are recovered from complementary slackness on points where beta_k > 0
(zero slack) and alpha_k > 0 (on the margin), respectively.
"""

import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False


def rbf_kernel(A, B, gamma):
    a_sq = np.sum(A ** 2, axis=1)[:, None]
    b_sq = np.sum(B ** 2, axis=1)[None, :]
    sqdist = a_sq + b_sq - 2 * A @ B.T
    np.maximum(sqdist, 0, out=sqdist)
    return np.exp(-gamma * sqdist)


def median_heuristic_gamma(X, n_pairs=2000, random_state=None):
    """gamma = 1 / (2 * median squared pairwise distance), estimated from a
    random sample of pairs — a standard, data-driven starting point for an
    RBF kernel bandwidth."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    sqdist = np.sum((X[i] - X[j]) ** 2, axis=1)
    sqdist = sqdist[sqdist > 0]
    med = np.median(sqdist) if len(sqdist) else 1.0
    return 1.0 / (2.0 * med)


class SVMPlus:
    """RBF-kernel SVM+ classifier. Labels must be in {-1, +1}."""

    def __init__(self, C=1.0, gamma=1.0, C_star=1.0, gamma_star=1.0, tol=1e-5):
        self.C = C
        self.gamma = gamma
        self.C_star = C_star
        self.gamma_star = gamma_star
        self.tol = tol

    def fit(self, X, X_star, y):
        n = len(y)
        y = np.asarray(y, dtype=float)
        if not set(np.unique(y)) <= {-1.0, 1.0}:
            raise ValueError("SVMPlus labels must be in {-1, +1}")

        K  = rbf_kernel(X, X, self.gamma)
        Ks = rbf_kernel(X_star, X_star, self.gamma_star)
        YKY = np.outer(y, y) * K
        Kc = Ks / self.C_star

        P = np.zeros((2 * n, 2 * n))
        P[:n, :n] = YKY + Kc
        P[:n, n:] = Kc
        P[n:, :n] = Kc
        P[n:, n:] = Kc
        P += 1e-8 * np.eye(2 * n)   # numerical jitter to keep cvxopt happy

        ones = np.ones(n)
        Ks1 = Ks @ ones
        q = np.concatenate([-ones - (self.C / self.C_star) * Ks1,
                             -(self.C / self.C_star) * Ks1])

        G = -np.eye(2 * n)
        h = np.zeros(2 * n)

        A = np.zeros((2, 2 * n))
        A[0, :n] = y
        A[1, :n] = 1.0
        A[1, n:] = 1.0
        b_eq = np.array([0.0, n * self.C])

        sol = cvxopt.solvers.qp(
            cvxopt.matrix(P), cvxopt.matrix(q),
            cvxopt.matrix(G), cvxopt.matrix(h),
            cvxopt.matrix(A), cvxopt.matrix(b_eq),
        )
        z = np.array(sol['x']).flatten()
        alpha, beta = z[:n].copy(), z[n:].copy()
        alpha[alpha < self.tol] = 0.0
        beta[beta < self.tol] = 0.0

        # Recover b* from points with zero slack (beta_k > 0), then xi_k for all k.
        u = alpha + beta - self.C
        Ksu = Ks @ u
        margin_star = beta > self.tol
        if not margin_star.any():
            margin_star = beta > 0
        b_star = -np.mean(Ksu[margin_star] / self.C_star) if margin_star.any() else 0.0
        xi = np.maximum(Ksu / self.C_star + b_star, 0.0)

        # Recover b from points on the decision margin (alpha_k > 0).
        sv = alpha > self.tol
        if not sv.any():
            sv = alpha > 0
        f_no_b = (alpha * y) @ K
        b = np.mean(y[sv] * (1 - xi[sv]) - f_no_b[sv]) if sv.any() else 0.0

        self.X_train_ = X
        self.y_train_ = y
        self.alpha_ = alpha
        self.b_ = b
        self.n_support_ = int(sv.sum())
        return self

    def decision_function(self, X):
        K = rbf_kernel(X, self.X_train_, self.gamma)
        return K @ (self.alpha_ * self.y_train_) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def predict_proba_pos(self, X):
        """Logistic squashing of the decision function, for AUC/PR curves.
        Not a calibrated probability — Platt scaling is not fit here."""
        d = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
