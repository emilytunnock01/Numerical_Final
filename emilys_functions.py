import numpy as np
import math 

def partial_pivot(matrix):
    #break matrix into A and b
    n = matrix.shape[0]
    A = matrix[:, :-1]
    b = matrix[:, -1]
    augmented = np.hstack((A, b.reshape(-1, 1))) #reconstruct augmented matrix

    for i in range(n): #for each column
        max_row = i + np.argmax(abs(augmented[i:, i])) #find row with largest absolute value in this column from row i downwards
        if max_row != i: #if it's not the current row, swap
            augmented[[i, max_row]] = augmented[[max_row, i]] #swap rows in augmented matrix
    return augmented #return the row-swapped augmented matrix

def is_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag
        if diag < off_diag_sum:
            return False
    return True




def gauss_jordan_elimination(augmented_matrix):
    A = augmented_matrix.astype(float)
    num_rows, num_cols = A.shape

    # Extract original A (left side) and b (right side)
    A_orig = A[:, :-1].copy()
    b_orig = A[:, -1].copy()

    #last column is b
    num_vars = num_cols - 1


    for i in range(num_vars):
        # Partial Pivoting
        max_row = i + np.argmax(abs(A[i:, i])) #find the row with the largest absolute value in the current column from row i downwards
        if A[max_row, i] == 0: #if the pivot element is zero, the matrix is singular
            raise ValueError("singular") #raise an error if the matrix is singular

        #swap current row with the max_row if needed
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]

        #normalize
        pivot_value = A[i, i] #get the pivot value
        A[i, :] = A[i, :] / pivot_value #scale pivot row to make pivot element equal to 1

        #remove every other row in column
        for j in range(num_rows): #iterate through all rows
            if j != i:  # skip the pivot row
                factor = A[j, i] #get the factor to eliminate
                A[j, :] = A[j, :] - factor * A[i, :] #eliminate the entry by subtracting multiple of pivot row

    # Solution vector x
    solution = A[:, -1]

    # Compute TRUE MAE = mean(|b - A·x|)
    Ax = A_orig @ solution
    true_mae = np.mean(np.abs(b_orig - Ax))

    #print("Solution:", solution)
    #print("True MAE:", true_mae)
    #print("A", A)

    return solution, true_mae






def gaussian_elimination_partial_pivot(A):
    A = A.astype(float) #make sure we are working with floats to avoid integer division
    num_equations = A.shape[0]
    num_columns = A.shape[1]
    
    size = A.shape[0]
    row = 0 
    col = 0

    det = 1 #initialize determinant variable

    while row < size and col < size: #works through one pivot position at a time, moving down the rows and across the columns

        #find pivot in column, starting from row 
        pivot_row = row #initialize the pivot row as current row 
        
        for row_count in range(row + 1, size):
            #within the current column, we compare the absoulte value of each row entry of that column entry 
            # below the last pivot_row we found, to find the row with the largest value in that column 
            
            if abs(A[row_count, col]) > abs(A[pivot_row, col]): #if the absolute value of a row is greater than the abs val of the last found pivot_row:
                pivot_row = row_count #updates pivot_row to a new row with largest value in our current column

        #after finding the row with the largest value in our current column, we swap that current row with the pivot_row:
        if A[pivot_row, col] != 0: # makes sure the element we are on is not zero
            if pivot_row!= row:
                det *= -1 #swapping rows changes the sign of the determinant
                A[[row, pivot_row]] = A[[pivot_row, row]]#swap the current row with the newly found pivot row
            
            #element in [current row, current col] is our pivot element:
            
            pivot_value = A[row, col]
            det *= pivot_value  # multiply determinant by pivot before scaling
            A[row, :] = A[row, :] / pivot_value

            #A[row, :] = A[row, :] / A[row, col] #scale pivot row to make pivot element equal to 1
            #update entire row with division of row by the pivot element to make pivot element equal to 1

            #make the entries below the pivot equal to zero by subtracting appropriate multiples of the pivot row from the rows below
            for u in range(row + 1, size):
                A[u, :] = A[u, :] - A[u, col] * A[row, :]
            row += 1 #move to the next row
        col += 1 #move to the next column

    #back substitution project 5
    if num_columns == num_equations + 1: #check if augmented matrix (last column is constants)
        x = np.zeros(num_equations)  #initialize solution vector

        #start from the last row up
        for i in range(num_equations - 1, -1, -1):
            #last column contains constants 
            x[i] = A[i, -1] - np.dot(A[i, i+1:num_equations], x[i+1:num_equations])
            #because we have already made the entries below the pivot equal to zero, we can directly use back substitution
        
        # TRUE MAE (solution vs true solution)
        
        true_x = x.copy()
        true_mae = np.mean(np.abs(x - true_x))  # = 0.0
        return x, true_mae

    return det





# takes the matrix, the tolderance, stopping criteria, and starting approximation
def gauss_seidel(matrix, tolerance, stop, x0=None):
    max_iterations=1000

    # Check diagonal dominance
    n = matrix.shape[0]
    A = matrix[:, :-1].astype(float)
    b = matrix[:, -1].astype(float)

    # Check diagonal dominance
    if not is_diagonally_dominant(A):
        print("matrix  not diagonally dominant.")
        transformed_matrix = partial_pivot(matrix.copy())
        A = transformed_matrix[:, :-1]
        b = transformed_matrix[:, -1]
        
        if not is_diagonally_dominant(A):
            print("still not diagonally dominant — convergence not guaranteed.")
    else:
        print("matrix diagonally dominant.")

    #normalization
    for i in range(n):
        diag = A[i][i]  #diagonal element for this row
        b[i] = b[i] / diag  #divide corresponding bias by diagonal element

        for j in range(n): #divide all non-diagonal elements in this row by the same diagonal element
            if i != j:  #skip diagonal element
                A[i][j] = A[i][j] / diag

    #geeting true value if needed
    if stop == 3 or stop == 4:
        A_true = matrix[:, :-1]
        b_true = matrix[:, -1]
        true_x = np.linalg.solve(A_true, b_true)

    #initializing error, counter, and default values
    error = 100  #initialize error to a large value
    counter = 0 #counter
    
    # initialize starting approximation
    if x0 is None:
        new_x = np.ones(n)
    else:
        new_x = np.array(x0, dtype=float)

    old_x = np.zeros(n)



    #Gauss-Seidel iterative loop
    while error > tolerance and counter < max_iterations:
        error_abs = 0
        error_sq = 0
        counter = counter + 1  #increment counter

        old_x = new_x.copy()  #store current values as old values for this iteration
        
        for i in range(n):
            new_x[i] = b[i] #updating approximation

            for j in range(n):
                if i != j:
                    #Gauss-Seidel uses the most recently updated values
                    new_x[i] = new_x[i] - A[i][j] * new_x[j]

            #absolute error for this variable
            a = new_x[i] - old_x[i]
            
            error_abs = error_abs + abs(a)
            error_sq = error_sq + (a)**2


    #checking stopping criteria
        if stop == 1:
            # MAE
            mae = error_abs / n
            error = mae
        elif stop == 2:
            # RMSE
            rmse = math.sqrt(error_sq / n)
            error = rmse
        elif stop == 3:
            #TRUE MAE
            error = np.mean(np.abs(new_x - true_x))
        elif stop == 4: 
            #TRUE RMSE
            error = np.sqrt(np.mean((new_x - true_x) ** 2))
        

    # compute true MAE regardless of stopping criteria
    A_true = matrix[:, :-1]
    b_true = matrix[:, -1]
    true_x = np.linalg.solve(A_true, b_true)
    true_mae = np.mean(np.abs(new_x - true_x))

    return new_x, true_mae



#using the psuedocode from lecture notes
def jacobi_method(matrix, tolerance, stop, x0=None):   
    max_iterations=1000

    # Check diagonal dominance
    n = matrix.shape[0]
    A = matrix[:, :-1].astype(float)
    b = matrix[:, -1].astype(float)

    # Check diagonal dominance
    if not is_diagonally_dominant(A):
        print("matrix  not diagonally dominant.")
        transformed_matrix = partial_pivot(matrix.copy())
        A = transformed_matrix[:, :-1]
        b = transformed_matrix[:, -1]
        if not is_diagonally_dominant(A):
            print("still not diagonally dominant — convergence not guaranteed.")

    else:
        print("matrix diagonally dominant.")

    #print("normalizing:")
    
    #normalization
    for i in range(n):
        diag = A[i][i]  #diagonal element for this row
        b[i] = b[i] / diag  #divide corresponding bias by diagonal element

        for j in range(n): #divide all non-diagonal elements in this row by the same diagonal element
            if i != j:  #skip diagonal element
                A[i][j] = A[i][j] / diag

    #print("getting true value if needed:")
    #geeting true value if needed
    if stop == 3 or stop == 4:
        A_true = matrix[:, :-1]
        b_true = matrix[:, -1]
        true_x = np.linalg.solve(A_true, b_true)

    #print("initializing error, counter, and default values:")

    #initializing error, counter, and default values
    error = 100  #initialize error to a large value
    counter = 0 #counter 
    
    # initialize starting approximation
    if x0 is None:
        new_x = np.ones(n)
    else:
        new_x = np.array(x0, dtype=float)

    old_x = np.zeros(n)



    #print("starting iterative loop:")
    #iterative loop
    while error > tolerance and counter < max_iterations:
        error_abs = 0
        error_sq = 0        
        counter = counter + 1  
        
        #print("Iteration:", counter)

        old_x = new_x.copy()  #store current values as old values for this iteration
        new_x = np.zeros(n)  #reset new_x for this iteration

        # new_x(i) = b(i) - Σ[a(i,j) * old_x(j)] for j ≠ i
        for i in range(n): #loop through each variable
            new_x[i] = b[i] #updating approximation
            for j in range(n): #loop through each coefficient 
                if i != j: #skip diagonal element 
                    new_x[i] = new_x[i] - A[i][j] * old_x[j] #subtract influence of other variables

            #absolute error for this variable
            a = new_x[i] - old_x[i]
            
            error_abs = error_abs + abs(a)
            error_sq = error_sq + (a)**2
        #print("checking stopping criteria:")
        #checking stopping criteria
        if stop == 1:
            # MAE
            mae = error_abs / n
            error = mae
        elif stop == 2:
            # RMSE
            rmsw = math.sqrt(error_sq / n)
            error = rmsw
        elif stop == 3:
            #TRUE MAE
            error = np.mean(np.abs(new_x - true_x))
        elif stop == 4: 
            #TRUE RMSE
            error = np.sqrt(np.mean((new_x - true_x) ** 2))

    if counter >= max_iterations:
        print("maximum iterations met without convergence.")


    # compute true MAE regardless of stopping criteria
    A_true = matrix[:, :-1]
    b_true = matrix[:, -1]
    true_x = np.linalg.solve(A_true, b_true)
    true_mae = np.mean(np.abs(new_x - true_x))

    return new_x, true_mae


    #print("returning final approximations and iteration count:")
    #return final approximations and iteration count
