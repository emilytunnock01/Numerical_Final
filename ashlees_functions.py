import math
import math
import copy
import numpy as np



def gaussian_elimination_pp(aug_matrix):
   
    A = copy.deepcopy(aug_matrix)
    n = len(A)  # number of equations

    # forward elimination with partial pivoting
    for i in range(n - 1):
       
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



def gauss_jordan_pp(aug_matrix):
 
    M = copy.deepcopy(aug_matrix)
    n = len(M)        
    m = len(M[0])     

    for i in range(n):
        
        pivot_row = i
        for r in range(i, n):
            if abs(M[r][i]) > abs(M[pivot_row][i]):
                pivot_row = r

        if abs(M[pivot_row][i]) < 1e-15:
            raise ValueError("No unique solution: zero pivot encountered.")

        
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
    return x



def is_diagonally_dominant(A):
   
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag < off:
            return False
    return True


def enforce_diag_dominance(aug_matrix):
   
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



def jacobi_iter(aug_matrix, tol=1e-6, max_iter=1000):
   
    n = len(aug_matrix)
    original = copy.deepcopy(aug_matrix)  
   

    A_rows = [row[:] for row in aug_matrix]
    A_only = [row[:n] for row in A_rows]
    if not is_diagonally_dominant(A_only):
        transformed = enforce_diag_dominance(A_rows)
        if transformed is not None:
            A_rows = transformed

 
    A = [row[:n] for row in A_rows]
    b = [row[n] for row in A_rows]

   
    for i in range(n):
        diag = A[i][i]
        b[i] /= diag
        for j in range(n):
            if i != j:
                A[i][j] /= diag

    x_true = gaussian_elimination_pp(original)

    x_old = [0.0] * n
    x_new = [1.0] * n

    iterations = 0
    approx_rmse = float("inf")
    approx_mae = float("inf")

    while approx_rmse > tol and iterations < max_iter:
        iterations += 1

       
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x_old[j]
            x_new[i] = b[i] - s

      
        diffs = [x_new[i] - x_old[i] for i in range(n)]
        approx_mae = sum(abs(d) for d in diffs) / n
        approx_rmse = math.sqrt(sum(d * d for d in diffs) / n)

        
        x_old = x_new[:]

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




def gauss_seidel_iter(aug_matrix, tol=1e-6, max_iter=1000):
   
    n = len(aug_matrix)
    original = copy.deepcopy(aug_matrix)

    
    A_rows = [row[:] for row in aug_matrix]
    A_only = [row[:n] for row in A_rows]
    if not is_diagonally_dominant(A_only):
        transformed = enforce_diag_dominance(A_rows)
        if transformed is not None:
            A_rows = transformed

 
    A = [row[:n] for row in A_rows]
    b = [row[n] for row in A_rows]

   
    for i in range(n):
        diag = A[i][i]
        b[i] /= diag
        for j in range(n):
            if i != j:
                A[i][j] /= diag

    
    x_true = gaussian_elimination_pp(original)

  
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
                    s += A[i][j] * x[j]  
            new_val = b[i] - s
            x_prev[i] = x[i]
            x[i] = new_val

       
        diffs = [x[i] - x_prev[i] for i in range(n)]
        approx_mae = sum(abs(d) for d in diffs) / n
        approx_rmse = math.sqrt(sum(d * d for d in diffs) / n)

  
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

