import tkinter as tk
from tkinter import ttk

def create_scenario_form(frame, scenario_number):
    """Creates a form for a single scenario inside the given frame."""
    scenario_frame = ttk.LabelFrame(frame, text=f"Scenario {scenario_number}", padding=(10, 10))
    scenario_frame.pack(fill='x', expand=True, padx=10, pady=5)

    # Orientation
    orientation_var = tk.StringVar(value='Vertical')
    ttk.Label(scenario_frame, text="Orientation:").grid(row=0, column=0, sticky='w')
    ttk.Radiobutton(scenario_frame, text="Vertical", variable=orientation_var, value="Vertical").grid(row=0, column=1, sticky='w')
    ttk.Radiobutton(scenario_frame, text="Horizontal", variable=orientation_var, value="Horizontal").grid(row=0, column=2, sticky='w')

    # Other parameters
    epochs_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Epochs:").grid(row=1, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=epochs_var, width=20).grid(row=1, column=1, sticky='w')

    batch_size_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Batch Size:").grid(row=2, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=batch_size_var, width=20).grid(row=2, column=1, sticky='w')

    num_clients_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Number of Clients:").grid(row=3, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_clients_var, width=20).grid(row=3, column=1, sticky='w')

    num_malicious_clients_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Number of Malicious Clients:").grid(row=4, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_malicious_clients_var, width=20).grid(row=4, column=1, sticky='w')

    attack_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Select Attack:").grid(row=5, column=0, sticky='w')
    ttk.Combobox(scenario_frame, textvariable=attack_var, values=('Attack 1', 'Attack 2', 'Attack 3'), width=18).grid(row=5, column=1, sticky='w')

    defense_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Select Defense:").grid(row=6, column=0, sticky='w')
    ttk.Combobox(scenario_frame, textvariable=defense_var, values=('Defense 1', 'Defense 2', 'Defense 3'), width=18).grid(row=6, column=1, sticky='w')

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

def run_simulation():
    # Placeholder for simulation logic
    results_text.set("Running simulations...")

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

# Scrollable frame to hold scenario forms
scenarios_frame = ttk.Frame(root)
scenarios_canvas = tk.Canvas(scenarios_frame)
scenarios_canvas.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(scenarios_frame, orient="vertical", command=scenarios_canvas.yview)
scrollbar.pack(side="right", fill="y")

scenarios_canvas.configure(yscrollcommand=scrollbar.set)
scenarios_canvas.bind('<Configure>', lambda e: scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all")))

scenarios_inner_frame = ttk.Frame(scenarios_canvas)
scenarios_canvas.create_window((0, 0), window=scenarios_inner_frame, anchor="nw")

scenarios_frame.pack(fill='both', expand=True, pady=10)

run_button = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_button.pack(pady=10)

results_text = tk.StringVar()
results_label = ttk.Label(root, textvariable=results_text)
results_label.pack(pady=10)

root.mainloop()