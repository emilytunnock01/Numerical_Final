import math
import copy

# -----------------------------
# 1. Gaussian elimination (direct) with partial pivoting
# -----------------------------

def gaussian_elimination_pp(aug_matrix):
    """
    Gaussian direct elimination with partial pivoting on an augmented matrix [A|b].

    aug_matrix : list of lists, shape n x (n+1)
    returns    : solution vector x as a list of floats
    """
    A = copy.deepcopy(aug_matrix)
    n = len(A)  # number of equations

    # forward elimination with partial pivoting
    for i in range(n - 1):
        # find pivot row in column i (max absolute value)
        pivot_row = i
        for r in range(i + 1, n):
            if abs(A[r][i]) > abs(A[pivot_row][i]):
                pivot_row = r

        if abs(A[pivot_row][i]) < 1e-15:
            raise ValueError("No unique solution: zero pivot encountered.")

        # swap current row with pivot row if needed
        if pivot_row != i:
            A[i], A[pivot_row] = A[pivot_row], A[i]

        # eliminate entries below the pivot
        for r in range(i + 1, n):
            factor = A[r][i] / A[i][i]
            for c in range(i, n + 1):
                A[r][c] -= factor * A[i][c]

    # check last pivot
    if abs(A[n - 1][n - 1]) < 1e-15:
        raise ValueError("No unique solution: zero pivot in last row.")

    # back substitution
    x = [0.0] * n
    x[n - 1] = A[n - 1][n] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (A[i][n] - s) / A[i][i]

    return x


# -----------------------------
# 2. Gauss-Jordan elimination with partial pivoting
# -----------------------------

def gauss_jordan_pp(aug_matrix):
    """
    Gauss-Jordan directed elimination with partial pivoting on [A|b].

    aug_matrix : list of lists, shape n x (n+1)
    returns    : (reduced_matrix, solution_vector)
                 reduced_matrix is row-reduced [I|x]
    """
    M = copy.deepcopy(aug_matrix)
    n = len(M)        # rows
    m = len(M[0])     # cols (should be n+1)

    for i in range(n):
        # partial pivoting in column i
        pivot_row = i
        for r in range(i, n):
            if abs(M[r][i]) > abs(M[pivot_row][i]):
                pivot_row = r

        if abs(M[pivot_row][i]) < 1e-15:
            raise ValueError("No unique solution: zero pivot encountered.")

        # swap pivot row into position
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]

        # normalize pivot row
        pivot = M[i][i]
        for c in range(i, m):
            M[i][c] /= pivot

        # eliminate all other entries in column i
        for r in range(n):
            if r == i:
                continue
            factor = M[r][i]
            if factor != 0.0:
                for c in range(i, m):
                    M[r][c] -= factor * M[i][c]

    # last column is the solution
    x = [M[i][m - 1] for i in range(n)]
    return M, x


# -----------------------------
# 3. Helpers for iterative methods
# -----------------------------

def is_diagonally_dominant(A):
    """
    Check if square matrix A is diagonally dominant by rows.
    A : list of lists (n x n)
    """
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag < off:
            return False
    return True


def enforce_diag_dominance(aug_matrix):
    """
    Try to reorder rows of augmented matrix [A|b] to make A diagonally dominant.
    Returns new augmented matrix or None if no such ordering is found.
    """
    A = copy.deepcopy(aug_matrix)
    n = len(A)
    used = set()
    new_rows = [None] * n

    for i in range(n):
        chosen = None
        for r in range(n):
            if r in used:
                continue
            diag = abs(A[r][i])
            off = sum(abs(A[r][j]) for j in range(n) if j != i)
            if diag >= off and diag > 0:
                chosen = r
                break
        if chosen is None:
            return None
        used.add(chosen)
        new_rows[i] = A[chosen][:]

    return new_rows


# -----------------------------
# 4. Jacobi iterative method
#    approximate + true MAE / RMSE
# -----------------------------

def jacobi_iter(aug_matrix, tol=1e-6, max_iter=1000):
    """
    Jacobi iterative method on [A|b].

    aug_matrix : list of lists, shape n x (n+1)
    tol        : stopping tolerance (based on approximate RMSE)
    max_iter   : maximum allowed iterations

    returns a dict:
        {
          'x'          : approximate solution vector,
          'iterations' : iteration count,
          'approx_mae' : MAE between successive iterates,
          'approx_rmse': RMSE between successive iterates,
          'true_mae'   : MAE vs true solution,
          'true_rmse'  : RMSE vs true solution
        }
    """
    n = len(aug_matrix)
    original = copy.deepcopy(aug_matrix)  # keep for "true" solution

    # try to enforce diagonal dominance
    A_rows = [row[:] for row in aug_matrix]
    A_only = [row[:n] for row in A_rows]
    if not is_diagonally_dominant(A_only):
        transformed = enforce_diag_dominance(A_rows)
        if transformed is not None:
            A_rows = transformed

    # rebuild A and b from (possibly) transformed system
    A = [row[:n] for row in A_rows]
    b = [row[n] for row in A_rows]

    # normalize each row to x_i = b_i - sum_{j≠i} a_ij x_j
    for i in range(n):
        diag = A[i][i]
        b[i] /= diag
        for j in range(n):
            if i != j:
                A[i][j] /= diag

    # "true" solution using direct Gaussian elimination
    x_true = gaussian_elimination_pp(original)

    # initial guess
    x_old = [0.0] * n
    x_new = [1.0] * n

    iterations = 0
    approx_rmse = float("inf")
    approx_mae = float("inf")

    while approx_rmse > tol and iterations < max_iter:
        iterations += 1

        # Jacobi update uses only previous iterate (x_old)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x_old[j]
            x_new[i] = b[i] - s

        # approximate error: difference between successive iterates
        diffs = [x_new[i] - x_old[i] for i in range(n)]
        approx_mae = sum(abs(d) for d in diffs) / n
        approx_rmse = math.sqrt(sum(d * d for d in diffs) / n)

        # prepare for next iteration
        x_old = x_new[:]

    # true error: difference vs exact solution
    true_diffs = [x_new[i] - x_true[i] for i in range(n)]
    true_mae = sum(abs(d) for d in true_diffs) / n
    true_rmse = math.sqrt(sum(d * d for d in true_diffs) / n)

    return {
        "x": x_new,
        "iterations": iterations,
        "approx_mae": approx_mae,
        "approx_rmse": approx_rmse,
        "true_mae": true_mae,
        "true_rmse": true_rmse,
    }


# -----------------------------
# 5. Gauss-Seidel iterative method
#    approximate + true MAE / RMSE
# -----------------------------

def gauss_seidel_iter(aug_matrix, tol=1e-6, max_iter=1000):
    """
    Gauss-Seidel iterative method on [A|b].

    aug_matrix : list of lists, shape n x (n+1)
    tol        : stopping tolerance (based on approximate RMSE)
    max_iter   : maximum allowed iterations

    returns a dict:
        {
          'x'          : approximate solution vector,
          'iterations' : iteration count,
          'approx_mae' : MAE between successive iterates,
          'approx_rmse': RMSE between successive iterates,
          'true_mae'   : MAE vs true solution,
          'true_rmse'  : RMSE vs true solution
        }
    """
    n = len(aug_matrix)
    original = copy.deepcopy(aug_matrix)

    # try to enforce diagonal dominance
    A_rows = [row[:] for row in aug_matrix]
    A_only = [row[:n] for row in A_rows]
    if not is_diagonally_dominant(A_only):
        transformed = enforce_diag_dominance(A_rows)
        if transformed is not None:
            A_rows = transformed

    # build A and b
    A = [row[:n] for row in A_rows]
    b = [row[n] for row in A_rows]

    # normalize each row
    for i in range(n):
        diag = A[i][i]
        b[i] /= diag
        for j in range(n):
            if i != j:
                A[i][j] /= diag

    # "true" solution from direct method
    x_true = gaussian_elimination_pp(original)

    # initial guess
    x = [1.0] * n
    x_prev = x[:]

    iterations = 0
    approx_rmse = float("inf")
    approx_mae = float("inf")

    while approx_rmse > tol and iterations < max_iter:
        iterations += 1

        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]  # uses latest values (Gauss-Seidel)
            new_val = b[i] - s
            x_prev[i] = x[i]
            x[i] = new_val

        # approximate error: successive iterates
        diffs = [x[i] - x_prev[i] for i in range(n)]
        approx_mae = sum(abs(d) for d in diffs) / n
        approx_rmse = math.sqrt(sum(d * d for d in diffs) / n)

    # true error vs exact solution
    true_diffs = [x[i] - x_true[i] for i in range(n)]
    true_mae = sum(abs(d) for d in true_diffs) / n
    true_rmse = math.sqrt(sum(d * d for d in true_diffs) / n)

    return {
        "x": x,
        "iterations": iterations,
        "approx_mae": approx_mae,
        "approx_rmse": approx_rmse,
        "true_mae": true_mae,
        "true_rmse": true_rmse,
    }

