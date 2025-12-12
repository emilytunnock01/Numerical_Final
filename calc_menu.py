import tkinter as tk
from tkinter import ttk, messagebox,filedialog, messagebox
import numpy as np

#this is outside the class so it can be used in load_matrix
def load_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            # skip empty lines
            if line.strip():
                # split by comma or whitespace
                if ',' in line:
                    row = [float(x) for x in line.strip().split(',')]
                else:
                    row = [float(x) for x in line.strip().split()]
                matrix.append(row)
    return np.array(matrix, dtype=float)


class MatrixCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Calculator")


    #SELECTOR FOR STUDENT'S CALCULATOR
        student = ["Ashlee", "Emily"]
        self.student = tk.StringVar(value=student[0])

        student_frame = tk.Frame(root)
        student_frame.pack(pady=10)

        tk.Label(student_frame, text="Student Calculator:").pack(side=tk.LEFT)
        ttk.Combobox(student_frame, textvariable=self.student, values=student, width=15).pack(side=tk.LEFT)

    #BUTTON TO LOAD MATRIX FROM FILE
        self.load_button = tk.Button(self.root, text="Load Matrix", command=self.load_matrix)
        self.load_button.pack()


    #MATRIX SIZE INPUTS
        self.rows = tk.IntVar(value=2)
        self.cols = tk.IntVar(value=2)

        size_frame = tk.Frame(root)
        size_frame.pack(pady=10)

        tk.Label(size_frame, text="Rows:").grid(row=0, column=0)
        tk.Entry(size_frame, textvariable=self.rows, width=5).grid(row=0, column=1)

        tk.Label(size_frame, text="Cols:").grid(row=0, column=2)
        tk.Entry(size_frame, textvariable=self.cols, width=5).grid(row=0, column=3)

        tk.Button(size_frame, text="Set Size", command=self.build_matrix_inputs).grid(row=0, column=4, padx=10)

    #MATRIX GRID PATTERN
        self.matrix_frame = tk.Frame(root)
        self.matrix_frame.pack(pady=10)
        self.inputs = []
        self.build_matrix_inputs()


    #METHOD DROPDOWN
        operations = ["Gauss Jordan", "Gauss Partial Pivot", "Gauss Seidel", "Jacobi"]
        self.operation = tk.StringVar(value=operations[0])

        op_frame = tk.Frame(root)
        op_frame.pack(pady=10)
        tk.Label(op_frame, text="Operation:").pack(side=tk.LEFT)

        self.op_combobox = ttk.Combobox(op_frame, textvariable=self.operation, values=operations, width=15)
        self.op_combobox.pack(side=tk.LEFT)
        self.op_combobox.bind('<<ComboboxSelected>>', self.on_operation_selected)


    #ITERATION PARAMETERS
        self.tolerance = tk.DoubleVar(value=0.001) #default tolerance
        self.stop = tk.StringVar(value="approximate MAE") #default stopping criteria

        #maps the stopping criteria to integer codes for the function calls
        self.stop_map = {
            "approximate MAE": 1,
            "approximate RMSE": 2,
            "true MAE": 3,
            "true RMSE": 4
        }

        self.starting_guess = None #will hold the starting guess as a list 
        self._iter_popup = None #reference to the popup window


    #CALCULATE BUTTON
        tk.Button(root, text="Calculate", command=self.calculate).pack(pady=10)

    #RESULT DISPLAY SECTION
        self.result_label = tk.Label(root, text="Result will appear here.")
        self.result_label.pack(pady=10)


##### MATRIX FROM GRID ###############
#BUILD THE MATRIX INPUT GRID
    def build_matrix_inputs(self):
        for widget in self.matrix_frame.winfo_children(): #for each existing widget in the frame
            widget.destroy() #remove it


        self.inputs = [] #reset inputs list 
        r = self.rows.get() #get the number of rows user typed
        c = self.cols.get()


        for i in range(r): #for each row in the matrix
            row_list = [] #list to hold the row's entries 
            for j in range(c): #for each column in the matrix 
                entry = tk.Entry(self.matrix_frame, width=5) #create an entry widget
                entry.grid(row=i, column=j, padx=5, pady=5) #place it in the grid
                row_list.append(entry) #add it to the row list
            self.inputs.append(row_list) #add the row list to the inputs list

# GET THE MATRIX FROM THE INPUT GRID AND RETURN AS 2D LIST
    def get_matrix(self):
        try:
            return [[float(self.inputs[i][j].get()) #get the value from each entry
                     for j in range(self.cols.get())] #for each column
                    for i in range(self.rows.get())]  #for each row
        except:
            messagebox.showerror("Error", "Matrix must contain valid numbers.") #if conversion fails
            return None



#### MATRIX LOAD FROM FILE ###############
    def load_matrix(self): #this function is called when load button is clicked
        filename = filedialog.askopenfilename( #open file dialog
            title="Select matrix file", 
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename: #if a file was selected
            try:
                self.mat = load_matrix_from_file(filename)  #load the matrix from the file by calling the function above
                # update the input grid to match the loaded matrix
                self.rows.set(self.mat.shape[0]) #set rows and cols to match loaded matrix
                self.cols.set(self.mat.shape[1]) 
                self.build_matrix_inputs() #rebuild the input grid
                # fill the entries with the loaded values
                for i in range(self.mat.shape[0]): #for each row
                    for j in range(self.mat.shape[1]): #for each column
                        self.inputs[i][j].delete(0, tk.END) #clear existing entry
                        self.inputs[i][j].insert(0, str(self.mat[i, j])) #insert loaded value
                messagebox.showinfo("matrix Loaded") 
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load matrix:\n{e}")



#POP UP FOR ITERATIVE METHOD PARAMETERS
    def on_operation_selected(self, event=None): #when operation is selected from dropdown
        op = self.operation.get() #get selected operation
        if op in ("Gauss Seidel", "Jacobi"): #if it's an iterative method
            self.open_iter_popup(op) #open the popup
        else:
            if self._iter_popup and tk.Toplevel.winfo_exists(self._iter_popup): #if popup exists
                self._iter_popup.destroy() #close it

    def open_iter_popup(self, op_name): #open the popup for iterative method parameters
        if self._iter_popup and tk.Toplevel.winfo_exists(self._iter_popup): #if popup already exists
            self._iter_popup.lift() #bring it to front 
            return

        popup = tk.Toplevel(self.root) #create new popup window 
        popup.title(f"{op_name} Parameters") #set title
        popup.transient(self.root) #set to be on top of main window
        popup.grab_set() #make it modal


    #TOLERANCE INPUT
        tk.Label(popup, text="Tolerance:").grid(row=0, column=0, padx=5, pady=5)
        tk.Entry(popup, textvariable=self.tolerance, width=20).grid(row=0, column=1, padx=5, pady=5)


        # STOPPING CRITERIA
        tk.Label(popup, text="Stopping Criteria:").grid(row=1, column=0, padx=5, pady=5)
        criteria_options = list(self.stop_map.keys())


        ttk.Combobox(
            popup, textvariable=self.stop, values=criteria_options, width=20
        ).grid(row=1, column=1, padx=5, pady=5)

    #STARTING APPROXIMATION INPUT
        tk.Label(popup, text="Starting Approximation (comma-separated):").grid(row=2, column=0, padx=5, pady=5)

        start_var = tk.StringVar() #to hold the starting guess string
        tk.Entry(popup, textvariable=start_var, width=25).grid(row=2, column=1, padx=5, pady=5) 

    #AUTO-GENERATE BUTTON
        def auto_generate():
            n = self.rows.get() #number of equations 
            start_var.set(",".join(["0"] * n)) #set starting guess to zeros

        tk.Button(popup, text="Auto Zero Vector", command=auto_generate).grid(row=3, column=0, columnspan=2, pady=5)


    #SAVE BUTTON
        def save_close():
            try:
                float(self.tolerance.get()) #validate tolerance
            except:
                messagebox.showerror("Error", "Invalid tolerance value.")
                return


        #parse starting approximation
            text = start_var.get().strip() #get the text typed
            if text:
                try: 
                    self.starting_guess_list = [float(x) for x in text.split(",")] #convert to list of floats
                except:
                    messagebox.showerror("Error", "Invalid starting approximation.") 
                    return
            


            popup.destroy() #close the popup


        tk.Button(popup, text="Save", command=save_close).grid(row=4, column=0, pady=10) #the save button
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=4, column=1, pady=10) #the cancel button


        self._iter_popup = popup #store reference to popup




#MATRIX CHECKS - PROJECT REQUIREMENTS
    def is_matrix_singular(self, matrix):
        import numpy as np
        try:
            np.linalg.inv(matrix)
            return False
        except:
            return True

    def is_diagonally_dominant(self, matrix):
        n = len(matrix)
        for i in range(n):
            row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
            if abs(matrix[i][i]) < row_sum:
                return False
        return True



#CALCULATION HANDLER
    def calculate(self):
        import numpy as np

        #get the selected student and operation
        student_name = self.student.get()
        mat = self.get_matrix()
        if mat is None:
            return
        mat = np.array(mat)
        op = self.operation.get()

        
    #check the matrix and provide warnings as needed
        # ITERATIVE METHODS NEED CHECKS
        if op in ("Gauss Seidel", "Jacobi") and not self.is_diagonally_dominant(mat):
            messagebox.showwarning("Warning", "Matrix is not diagonally dominant, convergence not gaurenteed.")


    #FUNCTION CALLS
        #FUNCTION CALLS
        try:            
            if student_name == "Ashlee":
            
                if op == "Gauss Jordan":
                    from ashlees_functions import gauss_jordan_pp
                    roots, TMAE = gauss_jordan_pp(mat)
                    roots = [float(r) for r in roots]
                    TMAE = float(TMAE)
                    result = f"Roots: {roots}\nTrue Mean Absolute Error: {TMAE}"
                
                    

                elif op == "Gauss Partial Pivot":
                    from ashlees_functions import gaussian_elimination_pp
                    roots, TMAE = gaussian_elimination_pp(mat)
                    roots = [float(r) for r in roots]
                    TMAE = float(TMAE)
                    result = f"Roots: {roots}\nTrue Mean Absolute Error: {TMAE}"

                elif op == "Gauss Seidel":
                    from ashlees_functions import gauss_seidel_iter
                    tol = float(self.tolerance.get())
                    stop_crit = self.stop_map[self.stop.get()]
                    x0 = np.array(self.starting_guess_list) if self.starting_guess_list else None
                    roots, TMAE = gauss_seidel_iter(mat, tol, stop=stop_crit, x0=x0)
                    roots = [float(r) for r in roots]
                    TMAE = float(TMAE)
                    result = f"Roots: {(roots)}\nTrue Mean Absolute Error: {TMAE}"

                elif op == "Jacobi":
                    from ashlees_functions import jacobi_iter
                    tol = float(self.tolerance.get())
                    stop_crit = self.stop_map[self.stop.get()]
                    x0 = np.array(self.starting_guess_list) if self.starting_guess_list else None
                    roots, TMAE = jacobi_iter(mat, tol, stop=stop_crit, x0=x0)
                    roots = [float(r) for r in roots]
                    TMAE = float(TMAE)
                    result = f"Roots: {(roots)}\nTrue Mean Absolute Error: {TMAE}"
                
                text = f"Result:\n{str(result)}"
                self.result_label.config(text=text)


        except Exception as e:
            messagebox.showerror("Matrix is singular no uniqe solution could be found", str(e))
        



        if student_name == "Emily":
            try:
                #print("tolerance:", type(self.tolerance))
                #print("stop:", type(self.stop))
                #print("starting_guess:", type(self.starting_guess))

                if hasattr(self, "mat") and self.mat is not None:
                    mat = self.mat
                else:
                    mat = np.array(self.get_matrix())




                #non iterative methods take only the matrix
                #they only return the roots and TMAE

                if op == "Gauss Jordan":
                    from emilys_functions import gauss_jordan_elimination
                    roots, TMAE = gauss_jordan_elimination(mat)

                elif op == "Gauss Partial Pivot":
                    from emilys_functions import gaussian_elimination_partial_pivot
                    roots, TMAE = gaussian_elimination_partial_pivot(mat)




                #iterative methods take matrix, tolerance, stopping criteria, and starting guess
                #they return roots and TMAE

                elif op == "Gauss Seidel":
                    from emilys_functions import gauss_seidel
                    
                    tol = float(self.tolerance.get())
                    stop_crit = self.stop_map[self.stop.get()]
                    x0 = np.array(self.starting_guess_list) if self.starting_guess_list else None # use the parsed list for the starting guess

                    roots, TMAE = gauss_seidel(mat, tol, stop_crit, x0=x0)


                elif op == "Jacobi":
                    from emilys_functions import jacobi_method
                    tol = float(self.tolerance.get())
                    stop_crit = self.stop_map[self.stop.get()]
                    x0 = np.array(self.starting_guess_list) if self.starting_guess_list else None
                    
                    roots, TMAE = jacobi_method(mat, tol, stop_crit, x0=x0)
                                    


                

                #output of the rusults into the result label
                self.result_label.config(
                    text=f"Roots: {roots}\nTrue Mean Absolute Error: {TMAE}"
                    ) 


            except Exception as e:
                    messagebox.showerror("Matrix is singular no uniqe solution could be found", str(e))








# --------------------------------------------
# Run Application
# --------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    MatrixCalculator(root)
    root.mainloop()