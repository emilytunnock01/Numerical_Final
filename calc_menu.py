import tkinter as tk
from tkinter import ttk, messagebox

# Matrix Calculator GUI using Tkinter
# -----------------------------------
# Features:
# - User chooses matrix size (rows & columns)
# - User enters all matrix values
# - Dropdown menu selects operation: Determinant, Transpose, Trace
# - Uses NumPy for calculations


class MatrixCalculator: #this is the physical window
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Calculator")

        #DROPDOWN TO SELECT WHOSE ALGORTITHMS TO BE USED
        student = ["Ashlee", "Emily"]
        self.student = tk.StringVar(value=student[0])
        student_frame = tk.Frame(root)
        student_frame.pack(pady=10)
        tk.Label(student_frame, text="Student Calculator:").pack(side=tk.LEFT)
        ttk.Combobox(student_frame, textvariable=self.student, values=student, width=15).pack(side=tk.LEFT)

        # Variables to store user-selected matrix size
        self.rows = tk.IntVar(value=2)
        self.cols = tk.IntVar(value=2)

        # --- MATRIX SIZE INPUTS ---
        size_frame = tk.Frame(root)
        size_frame.pack(pady=10)

        tk.Label(size_frame, text="Rows:").grid(row=0, column=0)
        tk.Entry(size_frame, textvariable=self.rows, width=5).grid(row=0, column=1)

        tk.Label(size_frame, text="Cols:").grid(row=0, column=2)
        tk.Entry(size_frame, textvariable=self.cols, width=5).grid(row=0, column=3)

        # Button rebuilds matrix input fields
        tk.Button(size_frame, text="Set Size", command=self.build_matrix_inputs).grid(row=0, column=4, padx=10)

        # Frame to hold matrix entry widgets
        self.matrix_frame = tk.Frame(root)
        self.matrix_frame.pack(pady=10)

        # Dropdown operations available
        operations = ["Gauss Jordan", "Gauss Partial Pivot", "Gauss Seidel", "Jacobi"]
        self.operation = tk.StringVar(value=operations[0])

        # Operation selection dropdown
        op_frame = tk.Frame(root)
        op_frame.pack(pady=10)

        tk.Label(op_frame, text="Operation:").pack(side=tk.LEFT)
        self.op_combobox = ttk.Combobox(op_frame, textvariable=self.operation, values=operations, width=15)
        self.op_combobox.pack(side=tk.LEFT)
        self.op_combobox.bind('<<ComboboxSelected>>', self.on_operation_selected)

        #FIX SCREEN SIZE, DONT HAVE
        #makes a popup for tolerance and stopping criteria if Gauss Seidel or Jacobi is selected
        self.tolerance = tk.DoubleVar(value=0.001)
        self.stop = tk.IntVar(value=1)
        self._iter_popup = None



        # Button to compute result
        tk.Button(root, text="Calculate", command=self.calculate).pack(pady=10)

        # Output label
        self.result_label = tk.Label(root, text="Result will appear here.")
        self.result_label.pack(pady=10)

        # List to store Entry widgets
        self.inputs = []

        # Create initial matrix input fields
        self.build_matrix_inputs()

    # --------------------------------------------
    # Create Entry widgets for matrix values
    # --------------------------------------------
    def build_matrix_inputs(self):
        # Clear old entry fields
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        self.inputs = []

        r = self.rows.get()
        c = self.cols.get()

        # Create a grid of Entry widgets
        for i in range(r):
            row_list = []
            for j in range(c):
                entry = tk.Entry(self.matrix_frame, width=5)
                entry.grid(row=i, column=j, padx=5, pady=5)
                row_list.append(entry)
            self.inputs.append(row_list)

    # --------------------------------------------
    # Convert Entry widgets to a 2D Python list
    # --------------------------------------------
    def get_matrix(self):
        try:
            return [[float(self.inputs[i][j].get()) for j in range(self.cols.get())] for i in range(self.rows.get())]
        except ValueError:
            messagebox.showerror("Error", "Matrix must contain numbers only.")
            return None
    

    #per the requirements we check if singular and diagonally dominant
    #NEED TO DO: HAVE PRINT STATEMENTS BE SEEN ON THE GUI
    def is_matrix_singular(self, matrix):
        import numpy as np
        try:
            np.linalg.inv(matrix)
            print("Matrix is not singular.")
            return False
        except np.linalg.LinAlgError:
            print("Matrix is singular.")
            return True
    
    def is_diagonally_dominant(self, matrix):
        n = len(matrix)
        for i in range(n):
            row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
            if abs(matrix[i][i]) < row_sum:
                print("WARNING: Matrix is not diagonally dominant.")
                return False
        print("Matrix is diagonally dominant.")
        return True

    def on_operation_selected(self, event=None):
        op = self.operation.get()
        if op in ("Gauss Seidel", "Jacobi"):
            self.open_iter_popup(op)
        else:
            # if popup exists, close it when not needed
            if self._iter_popup is not None and tk.Toplevel.winfo_exists(self._iter_popup):
                try:
                    self._iter_popup.destroy()
                except Exception:
                    self._iter_popup = None

    def open_iter_popup(self, op_name):
        # If already open, bring to front
        if self._iter_popup is not None and tk.Toplevel.winfo_exists(self._iter_popup):
            try:
                self._iter_popup.lift()
                return
            except Exception:
                self._iter_popup = None

        popup = tk.Toplevel(self.root)
        popup.title(f"{op_name} Parameters")
        popup.transient(self.root)
        popup.grab_set()

        tk.Label(popup, text="Tolerance:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        tol_entry = tk.Entry(popup, textvariable=self.tolerance, width=20)
        tol_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(popup, text="Stopping Criteria:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Combobox(popup, textvariable=self.stop, values=[1, 2, 3, 4], width=17).grid(row=1, column=1, padx=5, pady=5)

        btn_frame = tk.Frame(popup)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

        def save_and_close():
            try:
                _ = float(self.tolerance.get())
                _ = int(self.stop.get())
            except Exception:
                messagebox.showerror("Error", "Invalid tolerance or stopping criteria")
                return
            popup.destroy()

        tk.Button(btn_frame, text="Save", command=save_and_close).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=popup.destroy).pack(side=tk.LEFT, padx=5)

        self._iter_popup = popup

    #this is where we begin calculation
    def calculate(self):
        import numpy as np

        student_name = self.student.get()

        mat = self.get_matrix()
        if mat is None:
            return

        mat = np.array(mat)
        op = self.operation.get()

        #call to check
        is_matrix_singular = self.is_matrix_singular(mat)
        is_diagonally_dominant = self.is_diagonally_dominant(mat)

        #if student_name == "Ashlee":
        
        #use the files created by emily for the calculations
        if student_name == "Emily":
            try:
                if op == "Gauss Jordan":
                    from emilys_functions import gauss_jordan_elimination
                    result, solution = gauss_jordan_elimination(mat)
                elif op == "Gauss Partial Pivot":
                    from emilys_functions import gaussian_elimination_partial_pivot
                    result, solution = gaussian_elimination_partial_pivot(mat)
                elif op == "Gauss Seidel":
                    from emilys_functions import gauss_seidel
                    tol = float(self.tolerance.get())
                    stop_crit = int(self.stop.get())
                    result = gauss_seidel(mat, tol, stop_crit)
                elif op == "Jacobi":
                    from emilys_functions import jacobi_method
                    tol = float(self.tolerance.get())
                    stop_crit = int(self.stop.get())
                    result = jacobi_method(mat, tol, stop_crit)

                
                # Update UI with result
                self.result_label.config(text=f"Result:\n{result}")

            except Exception as e:
                messagebox.showerror("Error", str(e))


# --------------------------------------------
# Run Application
# --------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    MatrixCalculator(root)
    root.mainloop()