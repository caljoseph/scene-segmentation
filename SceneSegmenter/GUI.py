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
        self.filename_entry.insert(0, "./Data/Falling.txt")  # Default value

        # Create input boxes with labels and default values
        model_label = ttk.Label(root, text="Model:")
        model_label.grid(row=2, column=0)
        self.model_entry = ttk.Entry(root)
        self.model_entry.grid(row=2, column=1)
        self.model_entry.insert(0, 'all-MiniLM-L6-v2') 

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

        sigma_label = ttk.Label(root, text="Sigma:")
        sigma_label.grid(row=7, column=0)
        self.sigma_entry = ttk.Entry(root)
        self.sigma_entry.grid(row=7, column=1)
        self.sigma_entry.insert(0, 3)  # Default value

        # Create input boxes with labels and default values
        classifier_path_label = ttk.Label(root, text="Classifier Path:")
        classifier_path_label.grid(row=8, column=0)
        self.classifier_path_entry = ttk.Entry(root)
        self.classifier_path_entry.grid(row=8, column=1)
        self.classifier_path_entry.insert(0, './Classifiers/classifier_3_layer.pth') 

        # Create a Run button
        run_button = ttk.Button(root, text="Run", command=self.run_action)
        run_button.grid(row=9, column=1)

        # Create an Options button
        options_button = ttk.Button(root, text="Options", command=self.print_options_action)
        options_button.grid(row=10, column=0, sticky="W")

        # Start the GUI event loop
        root.mainloop()


    def run_action(self):
        # Retrieving data from the GUI elements
        plot_value = self.plot_var.get()
        filename_value = self.filename_entry.get()
        model_name_value = self.model_entry.get()
        split_method_value = self.split_method_entry.get()
        split_length_value = self.split_length_entry.get()
        smooth_value = self.smooth_entry.get()
        difference_measure_value = self.difference_measure_entry.get()
        sigma_value = self.sigma_entry.get()
        classifier_path = self.classifier_path_entry.get()
        
        #get inputs into right types
        if smooth_value == "none" or smooth_value == "None":
            smooth_value = None

        #get inputs into right types
        if classifier_path == "none" or classifier_path == "None":
            classifier_path = None

        split_length_value = int(split_length_value)
        sigma_value = float(sigma_value)
        

        
        try:
            self.function(filename=filename_value, 
                  split_method=split_method_value,
                  split_len = split_length_value,
                  smooth=smooth_value,
                  diff=difference_measure_value,
                  plot=plot_value,
                  sigma=sigma_value,
                  model_name=model_name_value,
                  classifier_path = classifier_path)
        except (TypeError, ValueError) as e:
            print(f"An error occurred: {e}")
        
        return plot_value, filename_value, split_method_value, split_length_value, smooth_value, difference_measure_value        
    
    def print_options_action(self):
        options = \
        {"filename" : [".txt (raw text)", ".csv (ground truth)"],
        "split_method (only used with txt files)": ["sentences", "tokens"],
        "smooth": ["gaussian1d", None],
        "diff": ["2norm"]}

        print("Options are as follows: ")
        for key in options.keys():
            print(f'{key}: {options[key]}')
        print("")

