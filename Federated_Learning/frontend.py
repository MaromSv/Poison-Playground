import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np




def create_scenario_form(frame, scenario_number):
    """Creates a form for a single scenario inside the given frame."""
    scenario_frame = ttk.LabelFrame(frame, text=f"Scenario {scenario_number}", padding=(10, 10))
    scenario_frame.pack(fill='x', expand=True, padx=10, pady=5)


    # Data Partitioning
    data_partitioning_var = tk.StringVar(value='Vertical')
    ttk.Label(scenario_frame, text="Data Partitioning:").grid(row=0, column=0, sticky='w')
    ttk.Radiobutton(scenario_frame, text="Vertical", variable=data_partitioning_var, value="Vertical").grid(row=0, column=1, padx=(0, 10), sticky='w')
    ttk.Radiobutton(scenario_frame, text="Horizontal", variable=data_partitioning_var, value="Horizontal").grid(row=0, column=2, padx=(0, 10), sticky='w')
    ttk.Radiobutton(scenario_frame, text="Horizontal - IID", variable=data_partitioning_var, value="Horizontal_IID").grid(row=0, column=3, padx=(0, 10), sticky='w')



    # Other parameters
    epochs_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Epochs:").grid(row=1, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=epochs_var, width=60).grid(row=1, column=1, columnspan=3, sticky='w')

    batch_size_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Batch Size:").grid(row=2, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=batch_size_var, width=60).grid(row=2, column=1, columnspan=3, sticky='w')
    num_clients_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Number of Clients:").grid(row=3, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_clients_var, width=60).grid(row=3, column=1, columnspan=3, sticky='w')

    num_malicious_clients_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Number of Malicious Clients:").grid(row=4, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_malicious_clients_var, width=60).grid(row=4, column=1, columnspan=3, sticky='w')

    attack_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Select Attack:").grid(row=5, column=0, sticky='w')
    ttk.Combobox(scenario_frame, textvariable=attack_var, values=('Attack 1', 'Attack 2', 'Attack 3'), width=57).grid(row=5, column=1, columnspan=3, sticky='w')

    defense_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Select Defense:").grid(row=6, column=0, sticky='w')
    ttk.Combobox(scenario_frame, textvariable=defense_var, values=('Defense 1', 'Defense 2', 'Defense 3'), width=57).grid(row=6, column=1, columnspan=3, sticky='w')

    # Update the scroll region after adding new widgets
    frame.update_idletasks()  # This updates the layout
    scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all"))

def create_scenarios():
    """Creates forms for the number of scenarios specified by the user."""
    try:
        num_scenarios = int(num_scenarios_var.get())
    except ValueError:
        results_text.set("Please enter a valid number of scenarios.")
        return

    # Clear previous scenario forms
    for widget in scenarios_frame.winfo_children():
        widget.destroy()

    # Create a form for each scenario
    for i in range(num_scenarios):
        create_scenario_form(scenarios_frame, i + 1)
        
    scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all"))

def run_simulation():
    # Example of how to use show_results function
    simulation_results = []

    for i in range(int(num_scenarios_var.get())):
        true_labels = np.random.randint(0, 10, 100)
        predicted_labels = np.random.randint(0, 10, 100)
        cm = confusion_matrix(true_labels, predicted_labels)
        simulation_results.append(cm)

    show_results(simulation_results)


def show_results(simulation_results):
    # Create a new window for results
    results_window = tk.Toplevel(root)
    results_window.title("Simulation Results")
    results_window.state('zoomed')  # Makes the window full screen

    # Create a frame to hold the plots
    plots_frame = ttk.Frame(results_window)
    plots_frame.pack(fill='both', expand=True)

    # Determine the number of rows and columns for the grid layout
    num_results = len(simulation_results)
    cols = 4  # You can adjust this based on how many columns you want
    rows = (num_results + cols - 1) // cols

    for index, simulation_result in enumerate(simulation_results):
        # Create a figure for the plot
        fig, ax = plt.subplots(figsize=(6, 4)) 
        sns.heatmap(simulation_result, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')



        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plots_frame)  # A tk.DrawingArea
        canvas_widget = canvas.get_tk_widget()
        # Calculate row and column index for grid placement
        row_index = index // cols
        col_index = index % cols
        canvas_widget.grid(row=row_index, column=col_index, padx=10, pady=20)
        canvas.draw()

    # Adjust the scrollbar to the new frame
    plots_frame.update_idletasks()
    plots_frame_width = plots_frame.winfo_width()
    plots_frame_height = plots_frame.winfo_height()
    plots_frame.pack_propagate(0)
    plots_frame.config(width=plots_frame_width, height=plots_frame_height)
    results_window.update_idletasks()
    results_window_width = results_window.winfo_width()
    results_window_height = results_window.winfo_height()
    results_window.geometry(f"{results_window_width}x{results_window_height}")






# Main window setup
root = tk.Tk()
root.title("PoisonPlayground - Federated Learning Simulator")
root.geometry("700x600")

# Number of scenarios input
num_scenarios_var = tk.StringVar()
num_scenarios_label = ttk.Label(root, text="Number of Scenarios:")
num_scenarios_label.pack(pady=5)
num_scenarios_entry = ttk.Entry(root, textvariable=num_scenarios_var)
num_scenarios_entry.pack(pady=5)

# Button to create scenario forms
create_scenarios_button = ttk.Button(root, text="Create Scenarios", command=create_scenarios)
create_scenarios_button.pack(pady=10)

# Scrollable frame setup
main_frame = ttk.Frame(root)  # Main frame to hold the canvas and scrollbar
main_frame.pack(fill='both', expand=True, pady=10)

scenarios_canvas = tk.Canvas(main_frame)
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=scenarios_canvas.yview)
scenarios_canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
scenarios_canvas.pack(side="left", fill="both", expand=True)

scenarios_frame = ttk.Frame(scenarios_canvas)  # This is the frame we will add widgets to
scenarios_canvas.create_window((0, 0), window=scenarios_frame, anchor="nw")

# This ensures that the canvas frame resizes to fit the inner frame
scenarios_canvas.bind('<Configure>', lambda e: scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all")))

run_button = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_button.pack(pady=10)

results_text = tk.StringVar()
results_label = ttk.Label(root, textvariable=results_text)
results_label.pack(pady=10)






root.mainloop()