import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    simulation_results = [
        {'data': [25, 30, 35, 40, 45], 'labels': ["Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5"]},
        {'data': [10, 20, 30, 40, 50], 'labels': ["Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5"]}
    ]
    show_results(simulation_results)


def show_results(simulation_results):
    # Create a new window for results
    results_window = tk.Toplevel(root)
    results_window.title("Simulation Results")
    results_window.geometry("800x600")
    for simulation_result in simulation_results:
        # Extract data for the current simulation result
        data = simulation_result['data']
        labels = simulation_result['labels']

        # Creating a figure for the plot
        fig, ax = plt.subplots(figsize=(4, 3)) 
        ax.plot(labels, data)

        # Adding figure to tkinter window
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='none', expand=False, side = "left")

        # # Example of displaying comparative numbers
        # ttk.Label(results_window, text="Comparative Numbers").pack()
        # for i, label in enumerate(labels):
        #     ttk.Label(results_window, text=f"{label}: {data[i]}").pack()





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