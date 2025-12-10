import tkinter as tk
from tkinter import ttk, messagebox


class MatrixCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Calculator")


        # ------------------------------------------------------
        # STUDENT DROPDOWN
        # ------------------------------------------------------
        student = ["Ashlee", "Emily"]
        self.student = tk.StringVar(value=student[0])


        student_frame = tk.Frame(root)
        student_frame.pack(pady=10)


        tk.Label(student_frame, text="Student Calculator:").pack(side=tk.LEFT)
        ttk.Combobox(student_frame, textvariable=self.student, values=student, width=15).pack(side=tk.LEFT)


        # ------------------------------------------------------
        # MATRIX SIZE INPUT
        # ------------------------------------------------------
        self.rows = tk.IntVar(value=2)
        self.cols = tk.IntVar(value=2)


        size_frame = tk.Frame(root)
        size_frame.pack(pady=10)


        tk.Label(size_frame, text="Rows:").grid(row=0, column=0)
        tk.Entry(size_frame, textvariable=self.rows, width=5).grid(row=0, column=1)


        tk.Label(size_frame, text="Cols:").grid(row=0, column=2)
        tk.Entry(size_frame, textvariable=self.cols, width=5).grid(row=0, column=3)


        tk.Button(size_frame, text="Set Size", command=self.build_matrix_inputs).grid(row=0, column=4, padx=10)


        # ------------------------------------------------------
        # MATRIX ENTRY GRID
        # ------------------------------------------------------
        self.matrix_frame = tk.Frame(root)
        self.matrix_frame.pack(pady=10)
        self.inputs = []
        self.build_matrix_inputs()


        # ------------------------------------------------------
        # OPERATION DROPDOWN
        # ------------------------------------------------------
        operations = ["Gauss Jordan", "Gauss Partial Pivot", "Gauss Seidel", "Jacobi"]
        self.operation = tk.StringVar(value=operations[0])


        op_frame = tk.Frame(root)
        op_frame.pack(pady=10)


        tk.Label(op_frame, text="Operation:").pack(side=tk.LEFT)


        self.op_combobox = ttk.Combobox(op_frame, textvariable=self.operation, values=operations, width=15)
        self.op_combobox.pack(side=tk.LEFT)
        self.op_combobox.bind('<<ComboboxSelected>>', self.on_operation_selected)


        # ------------------------------------------------------
        # ITERATIVE METHOD POPUP VARIABLES
        # ------------------------------------------------------
        self.tolerance = tk.DoubleVar(value=0.001)


        # STOPPING CRITERIA AS STRING
        self.stop = tk.StringVar(value="approximate MAE")


        # Text-to-number mapping
        self.stop_map = {
            "approximate MAE": 1,
            "approximate RMSE": 2,
            "true MAE": 3,
            "true RMSE": 4
        }


        self.starting_guess = None
        self._iter_popup = None


        # ------------------------------------------------------
        # CALCULATE BUTTON + RESULT LABEL
        # ------------------------------------------------------
        tk.Button(root, text="Calculate", command=self.calculate).pack(pady=10)


        self.result_label = tk.Label(root, text="Result will appear here.")
        self.result_label.pack(pady=10)


    # ------------------------------------------------------
    # BUILD MATRIX INPUT GRID
    # ------------------------------------------------------
    def build_matrix_inputs(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()


        self.inputs = []
        r = self.rows.get()
        c = self.cols.get()


        for i in range(r):
            row_list = []
            for j in range(c):
                entry = tk.Entry(self.matrix_frame, width=5)
                entry.grid(row=i, column=j, padx=5, pady=5)
                row_list.append(entry)
            self.inputs.append(row_list)


    # ------------------------------------------------------
    # GET MATRIX AS 2D FLOAT LIST
    # ------------------------------------------------------
    def get_matrix(self):
        try:
            return [[float(self.inputs[i][j].get())
                     for j in range(self.cols.get())]
                    for i in range(self.rows.get())]
        except:
            messagebox.showerror("Error", "Matrix must contain valid numbers.")
            return None


    # ------------------------------------------------------
    # MATRIX CHECKS
    # ------------------------------------------------------
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


    # ------------------------------------------------------
    # POPUP FOR TOLERANCE, STOPPING CRITERIA, STARTING GUESS
    # ------------------------------------------------------
    def on_operation_selected(self, event=None):
        op = self.operation.get()
        if op in ("Gauss Seidel", "Jacobi"):
            self.open_iter_popup(op)
        else:
            if self._iter_popup and tk.Toplevel.winfo_exists(self._iter_popup):
                self._iter_popup.destroy()


    def open_iter_popup(self, op_name):


        if self._iter_popup and tk.Toplevel.winfo_exists(self._iter_popup):
            self._iter_popup.lift()
            return


        popup = tk.Toplevel(self.root)
        popup.title(f"{op_name} Parameters")
        popup.transient(self.root)
        popup.grab_set()


        # TOLERANCE
        tk.Label(popup, text="Tolerance:").grid(row=0, column=0, padx=5, pady=5)
        tk.Entry(popup, textvariable=self.tolerance, width=20).grid(row=0, column=1, padx=5, pady=5)


        # STOPPING CRITERIA
        tk.Label(popup, text="Stopping Criteria:").grid(row=1, column=0, padx=5, pady=5)
        criteria_options = list(self.stop_map.keys())


        ttk.Combobox(
            popup, textvariable=self.stop, values=criteria_options, width=20
        ).grid(row=1, column=1, padx=5, pady=5)


        # STARTING APPROXIMATION INPUT
        tk.Label(popup, text="Starting Approximation (comma-separated):").grid(row=2, column=0, padx=5, pady=5)


        start_var = tk.StringVar()
        tk.Entry(popup, textvariable=start_var, width=25).grid(row=2, column=1, padx=5, pady=5)


        # AUTO-GENERATE BUTTON
        def auto_generate():
            n = self.rows.get()
            start_var.set(",".join(["0"] * n))


        tk.Button(popup, text="Auto Zero Vector", command=auto_generate).grid(row=3, column=0, columnspan=2, pady=5)


        # SAVE BUTTON
        def save_close():
            try:
                float(self.tolerance.get())
            except:
                messagebox.showerror("Error", "Invalid tolerance value.")
                return


            # Parse starting guess
            text = start_var.get().strip()
            if text:
                try:
                    self.starting_guess = [float(x) for x in text.split(",")]
                except:
                    messagebox.showerror("Error", "Invalid starting approximation.")
                    return
            else:
                self.starting_guess = None


            popup.destroy()


        tk.Button(popup, text="Save", command=save_close).grid(row=4, column=0, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=4, column=1, pady=10)


        self._iter_popup = popup


    # ------------------------------------------------------
    # MAIN CALCULATION
    # ------------------------------------------------------
    def calculate(self):
        import numpy as np


        student_name = self.student.get()
        mat = self.get_matrix()
        if mat is None:
            return


        mat = np.array(mat)
        op = self.operation.get()


        # SINGULAR CHECK
        if self.is_matrix_singular(mat):
            messagebox.showwarning("Warning", "Matrix may be singular.")


        # ITERATIVE METHODS NEED CHECKS
        if op in ("Gauss Seidel", "Jacobi") and not self.is_diagonally_dominant(mat):
            messagebox.showwarning("Warning", "Matrix is not diagonally dominant.")


        
        try:            
            if student_name == "Ashlee":
            
                if op == "Gauss Jordan":
                    from ashlees_functions import gauss_jordan_pp
                    solution = gauss_jordan_pp(mat)
                    solution = [float(x) for x in solution]
                    result = solution
                
                    

                elif op == "Gauss Partial Pivot":
                    from ashlees_functions import gaussian_elimination_pp
                    solution = gaussian_elimination_pp(mat)
                    solution = [float(x) for x in solution]
                    result = solution

                elif op == "Gauss Seidel":
                    from ashlees_functions import gauss_seidel_iter
                    tol = float(self.tolerance.get())
                    res = gauss_seidel_iter(mat, tol)
                    res["x"] = [float(x) for x in res["x"]]
                    # These are single values, not lists. No loop needed.
                    res["iterations"] = float(res["iterations"])
                    res["approx_mae"] = float(res["approx_mae"])
                    res["approx_rmse"] = float(res["approx_rmse"])
                    res["true_mae"] = float(res["true_mae"])
                    res["true_rmse"] = float(res["true_rmse"])
                    result = {
                        "Solution": res["x"],
                        "Iterations": res["iterations"],
                        "Approx MAE": res["approx_mae"],
                        "Approx RMSE": res["approx_rmse"],
                        "True MAE": res["true_mae"],
                        "True RMSE": res["true_rmse"],
                    }

                elif op == "Jacobi":
                    from ashlees_functions import jacobi_iter
                    tol = float(self.tolerance.get())
                    res = jacobi_iter(mat, tol)
                    res["x"] = [float(x) for x in res["x"]]
                    res["iterations"] = float(res["iterations"])
                    res["approx_mae"] = float(res["approx_mae"])
                    res["approx_rmse"] = float(res["approx_rmse"])
                    res["true_mae"] = float(res["true_mae"])
                    res["true_rmse"] = float(res["true_rmse"])
                    result = {
                        "Soltuion": res["x"],
                        "Iterations" : res["iterations"],
                        "Approx MAE": res["approx_mae"],
                        "Approx RMSE": res["approx_rmse"],
                        "True MAE": res["true_mae"],
                        "True RMSE": res["true_rmse"],
                    }
                
                text = f"Result:\n{result}"
                self.result_label.config(text=text)


        except Exception as e:
                messagebox.showerror("Error", str(e))
        
        #use the files created by emily for the calculations
        if student_name == "Emily":
            
            try:
                
                if op == "Gauss Jordan":
                    from emilys_functions import gauss_jordan_elimination
                    result, sol = gauss_jordan_elimination(mat)


                elif op == "Gauss Partial Pivot":
                    from emilys_functions import gaussian_elimination_partial_pivot
                    result, sol = gaussian_elimination_partial_pivot(mat)

                elif op == "Gauss Seidel":
                    from emilys_functions import gauss_seidel
                    tol = float(self.tolerance.get())
                    stop_crit = self.stop_map[self.stop.get()]
                    sol = gauss_seidel(mat, tol, stop_crit)


                elif op == "Jacobi":
                    from emilys_functions import jacobi_method
                    tol = float(self.tolerance.get())
                    stop_crit = self.stop_map[self.stop.get()]
                    sol = jacobi_method(mat, tol, stop_crit)


                self.result_label.config(text=f"{sol}")


            except Exception as e:
                    messagebox.showerror("Error", str(e))

                   


# --------------------------------------------
# Run Application
# --------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    MatrixCalculator(root)
    root.mainloop()