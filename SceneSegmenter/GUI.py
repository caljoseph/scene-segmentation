import tkinter as tk
from tkinter import ttk


class SE_GUI():
    def __init__(self, function):

        self.function = function

        # Set up the main application window
        root = tk.Tk()
        root.title("Parameter Input GUI")

        # Create a switch for the plot option
        self.plot_var = tk.BooleanVar(value=True)  # Default value set to True
        plot_checkbutton = ttk.Checkbutton(root, text="Plot", variable=self.plot_var, onvalue=True, offvalue=False)
        plot_checkbutton.grid(row=0, column=1)

        # Create input boxes with labels and default values
        filename_label = ttk.Label(root, text="Filename:")
        filename_label.grid(row=1, column=0)
        self.filename_entry = ttk.Entry(root)
        self.filename_entry.grid(row=1, column=1)
        self.filename_entry.insert(0, "Falling.txt")  # Default value


        split_method_label = ttk.Label(root, text="Split Method:")
        split_method_label.grid(row=3, column=0)
        self.split_method_entry = ttk.Entry(root)
        self.split_method_entry.grid(row=3, column=1)
        self.split_method_entry.insert(0, "sentences")  # Default value

        split_length_label = ttk.Label(root, text="Split Length:")
        split_length_label.grid(row=4, column=0)
        self.split_length_entry = ttk.Entry(root)
        self.split_length_entry.grid(row=4, column=1)
        self.split_length_entry.insert(0, "30")  # Default value

        smooth_label = ttk.Label(root, text="Smooth:")
        smooth_label.grid(row=5, column=0)
        self.smooth_entry = ttk.Entry(root)
        self.smooth_entry.grid(row=5, column=1)
        self.smooth_entry.insert(0, "gaussian1d")  # Default value

        difference_measure_label = ttk.Label(root, text="Difference Measure:")
        difference_measure_label.grid(row=6, column=0)
        self.difference_measure_entry = ttk.Entry(root)
        self.difference_measure_entry.grid(row=6, column=1)
        self.difference_measure_entry.insert(0, "2norm")  # Default value

        # Create a Run button
        run_button = ttk.Button(root, text="Run", command=self.run_action)
        run_button.grid(row=7, column=1)

        # Start the GUI event loop
        root.mainloop()


    def run_action(self):
        # Retrieving data from the GUI elements
        plot_value = self.plot_var.get()
        filename_value = self.filename_entry.get()
        split_method_value = self.split_method_entry.get()
        split_length_value = self.split_length_entry.get()
        smooth_value = self.smooth_entry.get()
        difference_measure_value = self.difference_measure_entry.get()

        #check inputs
        if split_method_value not in ["sentences", "tokens"]:
            raise ValueError(f'{split_method_value} is an invalid value for split_method, allowed values are: {["sentences", "tokens"]}')
        if difference_measure_value not in ["2norm"]:
            raise ValueError(f"{difference_measure_value} is invalid difference measure. Valid values are ['2norm']")
        if smooth_value not in ["gaussian1d", "none", None]:
            raise ValueError(f"{smooth_value} is invalid smoothing method. Valid values are ['gaussian1d', None]")
        
        #get inputs into right types
        if smooth_value == "none":
            smooth_value = None
        

        
        self.function(filename=filename_value, 
                  split_method=split_method_value,
                  smooth=smooth_value,
                  diff=difference_measure_value,
                  plot=plot_value)
        
        return plot_value, filename_value, split_method_value, split_length_value, smooth_value, difference_measure_value        