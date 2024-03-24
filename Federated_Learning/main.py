import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import threading
import time

from simulationVertical import runVerticalSimulation
from simulationHorizontal import runHorizontalSimulation

global simulationComplete
simulationComplete = False

global scenario_vars
scenario_vars = {} # This will store the variables for all scenarios
numOfClasses = 10


DEFAULT_VALUES = {
    "name": "Default Scenario",
    "data_partitioning": "Horizontal",  # Vertical, Horizontal, or Horizontal_IID
    "epochs": "3",
    "batch_size": "16",
    "num_clients": "2",
    "num_malicious_clients": "0",
    "attack": "None",  # None, Label Flipping, Model Poisoning, Watermark
    "defense": "None",  # None, Two_Norm, Fools Gold
}



def create_scenario_form(frame, scenario_number):
    """Creates a form for a single scenario inside the given frame."""
    scenario_frame = ttk.LabelFrame(frame, text=f"Scenario {scenario_number}", padding=(10, 10))
    scenario_frame.pack(fill='x', expand=True, padx=10, pady=5)
    
    global scenario_vars


    #Scenario name: 
    name = tk.StringVar()
    ttk.Label(scenario_frame, text="Name:").grid(row=0, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=name, width=60).grid(row=0, column=1, columnspan=3, sticky='w')
    scenario_vars[f"name_{scenario_number}"] = name


    # Data Partitioning
    data_partitioning_var = tk.StringVar(value=None)
    ttk.Label(scenario_frame, text="Data Partitioning:").grid(row=1, column=0, sticky='w')
    ttk.Radiobutton(scenario_frame, text="Vertical", variable=data_partitioning_var, value="Vertical").grid(row=1, column=1, padx=(0, 10), sticky='w')
    ttk.Radiobutton(scenario_frame, text="Horizontal", variable=data_partitioning_var, value="Horizontal").grid(row=1, column=2, padx=(0, 10), sticky='w')
    ttk.Radiobutton(scenario_frame, text="Horizontal - IID", variable=data_partitioning_var, value="Horizontal_IID").grid(row=1, column=3, padx=(0, 10), sticky='w')
    scenario_vars[f"data_partitioning_var_{scenario_number}"] = data_partitioning_var

    # Other parameters
    epochs_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Epochs:").grid(row=2, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=epochs_var, width=60).grid(row=2, column=1, columnspan=3, sticky='w')
    scenario_vars[f"epochs_var_{scenario_number}"] = epochs_var

    batch_size_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Batch Size:").grid(row=3, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=batch_size_var, width=60).grid(row=3, column=1, columnspan=3, sticky='w')
    scenario_vars[f"batch_size_var_{scenario_number}"] = batch_size_var

    num_clients_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Number of Clients:").grid(row=4, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_clients_var, width=60).grid(row=4, column=1, columnspan=3, sticky='w')
    scenario_vars[f"num_clients_var_{scenario_number}"] = num_clients_var



    num_malicious_clients_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Number of Malicious Clients:").grid(row=5, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_malicious_clients_var, width=60).grid(row=5, column=1, columnspan=3, sticky='w')
    scenario_vars[f"num_malicious_clients_var_{scenario_number}"] = num_malicious_clients_var


    attack_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Select Attack:").grid(row=6, column=0, sticky='w')
    attack_combobox = ttk.Combobox(scenario_frame, textvariable=attack_var, values=('None', 'Label Flipping', 'Model Poisoning', 'Single Pixel Attack'), width=57)
    attack_combobox.grid(row=6, column=1, columnspan=3, sticky='w')
    attack_combobox.bind("<<ComboboxSelected>>", lambda event, frame=scenario_frame: update_attack_config(frame, attack_var.get(), scenario_number, 7))
    scenario_vars[f"attack_var_{scenario_number}"] = attack_var


    defense_var = tk.StringVar()
    ttk.Label(scenario_frame, text="Select Defense:").grid(row=7, column=0, sticky='w')
    defence_combobox = ttk.Combobox(scenario_frame, textvariable=defense_var, values=('None', 'Two_Norm', 'Fools Gold'), width=57)
    defence_combobox.grid(row=7, column=1, columnspan=3, sticky='w')
    defence_combobox.bind("<<ComboboxSelected>>", lambda event, frame=scenario_frame: update_defence_config(frame, defense_var.get(), scenario_number, 7))
    scenario_vars[f"defense_var_{scenario_number}"] = defense_var


    scenario_vars[f"attackParams_{scenario_number}"] = []
    scenario_vars[f"defenceParams_{scenario_number}"] = []
    # Update the scroll region after adding new widgets
    frame.update_idletasks()  # This updates the layout
    scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all"))




def update_attack_config(scenario_frame, attack, scenario_number, rowNum):

    global scenario_vars

    # Clear previous attack configuration fields
    for widget in scenario_frame.winfo_children():
        widget.grid_forget()


    # # Call update_defence_config here
    # defence_var = scenario_vars[f"defense_var_{scenario_number}"].get()
    # update_defence_config(scenario_frame, defence_var, scenario_number, rowNum)

    #Scenario name: 
    name = scenario_vars[f"name_{scenario_number}"]
    ttk.Label(scenario_frame, text="Name:").grid(row=0, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=name, width=60).grid(row=0, column=1, columnspan=3, sticky='w')
    scenario_vars[f"name_{scenario_number}"] = name

    # Data Partitioning
    data_partitioning_var = scenario_vars[f"data_partitioning_var_{scenario_number}"]
    ttk.Label(scenario_frame, text="Data Partitioning:").grid(row=1, column=0, sticky='w')
    ttk.Radiobutton(scenario_frame, text="Vertical", variable=data_partitioning_var, value="Vertical").grid(row=1, column=1, padx=(0, 10), sticky='w')
    ttk.Radiobutton(scenario_frame, text="Horizontal", variable=data_partitioning_var, value="Horizontal").grid(row=1, column=2, padx=(0, 10), sticky='w')
    ttk.Radiobutton(scenario_frame, text="Horizontal - IID", variable=data_partitioning_var, value="Horizontal_IID").grid(row=1, column=3, padx=(0, 10), sticky='w')
    scenario_vars[f"data_partitioning_var_{scenario_number}"] = data_partitioning_var

    # Other parameters
    try:
        epochs_var = scenario_vars[f"epochs_var_{scenario_number}"]
    except:
        pass
    ttk.Label(scenario_frame, text="Epochs:").grid(row=2, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=epochs_var, width=60).grid(row=2, column=1, columnspan=3, sticky='w')
    scenario_vars[f"epochs_var_{scenario_number}"] = epochs_var


        
    batch_size_var = scenario_vars[f"batch_size_var_{scenario_number}"]
    ttk.Label(scenario_frame, text="Batch Size:").grid(row=3, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=batch_size_var, width=60).grid(row=3, column=1, columnspan=3, sticky='w')
    scenario_vars[f"batch_size_var_{scenario_number}"] = batch_size_var

    
    num_clients_var = scenario_vars[f"num_clients_var_{scenario_number}"]

    ttk.Label(scenario_frame, text="Number of Clients:").grid(row=4, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_clients_var, width=60).grid(row=4, column=1, columnspan=3, sticky='w')
    scenario_vars[f"num_clients_var_{scenario_number}"] = num_clients_var


 
    num_malicious_clients_var = scenario_vars[f"num_malicious_clients_var_{scenario_number}"]
 
    ttk.Label(scenario_frame, text="Number of Malicious Clients:").grid(row=5, column=0, sticky='w')
    ttk.Entry(scenario_frame, textvariable=num_malicious_clients_var, width=60).grid(row=5, column=1, columnspan=3, sticky='w')
    scenario_vars[f"num_malicious_clients_var_{scenario_number}"] = num_malicious_clients_var

    
    attack_var = tk.StringVar(value=attack)  # Update attack variable with the selected attack

    ttk.Label(scenario_frame, text="Select Attack:").grid(row=6, column=0, sticky='w')
    attack_combobox = ttk.Combobox(scenario_frame, textvariable=attack_var, values=('None', 'Label Flipping', 'Model Poisoning'), width=57)
    attack_combobox.grid(row=6, column=1, columnspan=3, sticky='w')
    # Isn't calling update_attack_config here recursive??????????????????????????????????????????????????????????????????????????????????????
    attack_combobox.bind("<<ComboboxSelected>>", lambda event, frame=scenario_frame: update_attack_config(frame, attack_var.get(), scenario_number, 7))
    scenario_vars[f"attack_var_{scenario_number}"] = attack_var

    



    attackParams = []
    if attack == 'Label Flipping':
        source_label_var = tk.StringVar()
        target_label_var = tk.StringVar()
        ttk.Label(scenario_frame, text="Source Label:").grid(row=8, column=0, sticky='w')
        ttk.Combobox(scenario_frame, textvariable=source_label_var, values=list(range(10)), width=57).grid(row=8, column=1, columnspan=3, sticky='w')
        ttk.Label(scenario_frame, text="Target Label:").grid(row=9, column=0, sticky='w')
        ttk.Combobox(scenario_frame, textvariable=target_label_var, values=list(range(10)), width=57).grid(row=9, column=1, columnspan=3, sticky='w')

        attackParams.append(source_label_var)
        attackParams.append(target_label_var)
        rowNum += 2
  
        
    elif attack == 'Model Poisoning':
        poisoning_scale_var = tk.DoubleVar()
        ttk.Label(scenario_frame, text="Poisoning Scale:").grid(row=8, column=0, sticky='w')
        ttk.Entry(scenario_frame, textvariable=poisoning_scale_var, width=60).grid(row=8, column=1, columnspan=3, sticky='w')
        attackParams.append(poisoning_scale_var)
        rowNum += 1

    elif attack == "Single Pixel Attack":
        scale_var = tk.DoubleVar()
        target_label_var = tk.StringVar()
        ttk.Label(scenario_frame, text="nb_iter:").grid(row=8, column=0, sticky='w')
        ttk.Entry(scenario_frame, textvariable=scale_var, width=60).grid(row=8, column=1, columnspan=3, sticky='w')
        ttk.Label(scenario_frame, text="eps:").grid(row=9, column=0, sticky='w')
        ttk.Combobox(scenario_frame, textvariable=target_label_var, values=list(range(10)), width=57).grid(row=9, column=1, columnspan=3, sticky='w')
        attackParams.append(scale_var)
        attackParams.append(target_label_var)
        rowNum += 2

    else:
        pass

    defense_var = scenario_vars[f"defense_var_{scenario_number}"]
    ttk.Label(scenario_frame, text="Select Defense:").grid(row=rowNum+1, column=0, sticky='w')
    defence_combobox = ttk.Combobox(scenario_frame, textvariable=defense_var, values=('None', 'Two_Norm', 'Fools Gold'), width=57)
    defence_combobox.grid(row=rowNum + 1, column=1, columnspan=3, sticky='w')
    defence_combobox.bind("<<ComboboxSelected>>", lambda event, frame=scenario_frame: update_defence_config(frame, defense_var.get(), scenario_number, rowNum))
    scenario_vars[f"defense_var_{scenario_number}"] = defense_var

    scenario_vars[f"attackParams_{scenario_number}"] = attackParams

    # Update the scroll region after adding new widgets
    scenario_frame.update_idletasks()
    scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all"))


    # Update the scroll region after adding new widgets
    scenario_frame.update_idletasks()
    scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all"))


def update_defence_config(scenario_frame, defence, scenario_number, rowNum):
    global scenario_vars

    # Clear existing defense configuration widgets if any
    for child in scenario_frame.winfo_children():
        if "defence_param" in child.winfo_name():
            child.destroy()

    # Adjusting the row number to insert the defense parameters below the defense selection
    defence_row = rowNum + 2  # Assuming defense combobox is at 'rowNum + 1'

    defenceParams = []
    if defence == 'Two_Norm':
        # Creating the UI components for Two_Norm defense parameters
        norm_threshold_var = tk.DoubleVar()
        norm_threshold_label = ttk.Label(scenario_frame, text="Norm Threshold:", name=f"defence_param_{scenario_number}_label")
        norm_threshold_entry = ttk.Entry(scenario_frame, textvariable=norm_threshold_var, width=60, name=f"defence_param_{scenario_number}_entry")

        # Positioning the newly created widgets
        norm_threshold_label.grid(row=defence_row, column=0, sticky='w')
        norm_threshold_entry.grid(row=defence_row, column=1, columnspan=3, sticky='w')
        defenceParams.append(norm_threshold_var)
        

    elif defence == 'Fools Gold':
        # Creating the UI components for Fools Gold defense parameters
        confidence_param_var = tk.DoubleVar()
        confidence_param_label = ttk.Label(scenario_frame, text="Confidence Param:", name=f"defence_param_{scenario_number}_label")
        confidence_param_entry = ttk.Entry(scenario_frame, textvariable=confidence_param_var, width=60, name=f"defence_param_{scenario_number}_entry")
        defenceParams.append(confidence_param_var)
        # Positioning the newly created widgets
        confidence_param_label.grid(row=defence_row, column=0, sticky='w')
        confidence_param_entry.grid(row=defence_row, column=1, columnspan=3, sticky='w')


    scenario_vars[f"defenceParams_{scenario_number}"] = defenceParams
    # Ensure the rest of the UI is adjusted accordingly
    scenario_frame.update_idletasks()
    scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all"))


def set_default_values():
    global scenario_vars
    for i in range(int(num_scenarios_var.get())):
        scenario_vars[f"name_{i}"].set(DEFAULT_VALUES["name"])
        scenario_vars[f"data_partitioning_var_{i}"].set(DEFAULT_VALUES["data_partitioning"])
        scenario_vars[f"epochs_var_{i}"].set(DEFAULT_VALUES["epochs"])
        scenario_vars[f"batch_size_var_{i}"].set(DEFAULT_VALUES["batch_size"])
        scenario_vars[f"num_clients_var_{i}"].set(DEFAULT_VALUES["num_clients"])
        scenario_vars[f"num_malicious_clients_var_{i}"].set(DEFAULT_VALUES["num_malicious_clients"])
        scenario_vars[f"attack_var_{i}"].set(DEFAULT_VALUES["attack"])
        scenario_vars[f"defense_var_{i}"].set(DEFAULT_VALUES["defense"])



def create_scenarios():

    """Creates forms for the number of scenarios specified by the user."""
    try:
        num_scenarios = int(num_scenarios_var.get())
    except ValueError:
        results_text.set("Please enter a valid number of scenarios.")
        return

    #Reset the variables
    global scenario_vars
    scenario_vars = {} 

    # Clear previous scenario forms
    for widget in scenarios_frame.winfo_children():
        widget.destroy()

    # Create a form for each scenario
    for i in range(num_scenarios):
        create_scenario_form(scenarios_frame, i)
        
    scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all"))



def checkAllFieldsFilled():
    global scenario_vars
    all_keys_list = list(scenario_vars.keys())

    for i, key in enumerate(all_keys_list):
        try:
            if scenario_vars[key].get() == "":
                return False
        except:
            pass
    
    return True


##TODO: GET THE LOADING BAR TO WORK PROPERLY
def start_simulation():
    if not checkAllFieldsFilled():
        results_text.set("Please fill in all fields")
        return
    # Show loading screen in a separate thread
    # threading.Thread(target=loadingScreen, daemon=True).start()
    
    run_simulation()
    # # Start simulation in another separate thread
    # threading.Thread(target=run_simulation, daemon=True).start()


def run_simulation():
    #Disable run button
    run_button.config(state="disabled")
    # Example of how to use show_results function
    simulation_results = []
    scenario_names = []
    for i in range(int(num_scenarios_var.get())):
        name = scenario_vars[f"name_{i}"].get()
        numEpochs = int(scenario_vars[f"epochs_var_{i}"].get())
        batchSize = int(scenario_vars[f"batch_size_var_{i}"].get())
        numClients = int(scenario_vars[f"num_clients_var_{i}"].get())
        numMalClients = int(scenario_vars[f"num_malicious_clients_var_{i}"].get() )
        attack = scenario_vars[f"attack_var_{i}"].get()
        defence = scenario_vars[f"defense_var_{i}"].get()
        attackParams = scenario_vars[f"attackParams_{i}"] 
        defenceParams = scenario_vars[f"defenceParams_{i}"]


        attackParamsList = []
        for j in range(len(attackParams)):
            attackParamsList.append(int(attackParams[j].get()))
      

        defenceParamsList = []
        for j in range(len(defenceParams)):
            defenceParamsList.append(int(defenceParams[j].get()))
   

      
        seed = int(seed_var.get())

        trials = int(num_trials_var.get())
        trialResults = []
        for j in range(trials):
            seed += j # Change the seed for every trial
            if scenario_vars[f"data_partitioning_var_{i}"].get() == "Vertical":

                accuracy, cm = runVerticalSimulation(numEpochs, batchSize, numClients, numMalClients, 
                                attack, defence, attackParamsList, defenceParamsList, seed)
                
            elif scenario_vars[f"data_partitioning_var_{i}"].get() == "Horizontal":

                accuracy, cm = runHorizontalSimulation(False, numEpochs, batchSize, numClients, numMalClients, 
                                attack, defence, attackParamsList, defenceParamsList, seed)
            
            trialResults.append(cm)
        
        cmSum = confusion_matrix(y_true=[], y_pred=[], labels=range(numOfClasses)) #(re)Initialize empty CM
        for m in trialResults:
            cmSum +=m

        
        simulation_results.append(cmSum)
        scenario_names.append(name)

    global simulationComplete
    simulationComplete = True
    
    show_results(simulation_results, scenario_names)

   
    # Enable the run button after simulations are complete
    run_button.config(state="normal")
    simulationComplete = False

def loadingScreen():
    # Create a loading screen
    loading_screen = tk.Toplevel(root)
    loading_screen.title("Loading...")
    loading_screen.geometry("300x100")
    loading_label = ttk.Label(loading_screen, text="Running simulations, please wait...")
    loading_label.pack(pady=10)
    progress_bar = ttk.Progressbar(loading_screen, mode="indeterminate")
    progress_bar.pack(pady=5)
    progress_bar.start()

    global simulationComplete
    while not simulationComplete: 
        loading_screen.update()

    loading_screen.destroy()


def show_results(simulation_results, scenario_names):
    # Create a new window for results
    results_window = tk.Toplevel(root)
    results_window.title("Simulation Results")
    results_window.state('zoomed')  # Makes the window full screen

    # Create a frame to hold the plots
    plots_frame = ttk.Frame(results_window)
    plots_frame.pack(fill='both', expand=True)

    # Create a canvas for the plots
    plots_canvas = tk.Canvas(plots_frame)
    plots_canvas.pack(side="left", fill="both", expand=True)

    # Add a scrollbar to the canvas
    scrollbar = ttk.Scrollbar(plots_frame, orient="vertical", command=plots_canvas.yview)
    scrollbar.pack(side="right", fill="y")
    plots_canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame inside the canvas to hold the plots
    plots_inner_frame = ttk.Frame(plots_canvas)
    plots_canvas.create_window((0, 0), window=plots_inner_frame, anchor="nw")

    # Determine the number of rows and columns for the grid layout
    num_results = len(simulation_results)
    cols = 2 # You can adjust this based on how many columns you want
    rows = (num_results + cols - 1) // cols

    accuracies = []  # List to store final accuracies

    for index, simulation_result in enumerate(simulation_results):
        # Calculate accuracy and append to accuracies list
        accuracy = np.sum(np.diag(simulation_result)) / np.sum(simulation_result)
        accuracies.append(accuracy)

        # Create a figure for the confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 4)) 
        sns.heatmap(simulation_result, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix for {scenario_names[index]}')

        # Embed the confusion matrix plot in the inner frame
        canvas = FigureCanvasTkAgg(fig, master=plots_inner_frame)
        canvas_widget = canvas.get_tk_widget()
        # Calculate row and column index for grid placement
        row_index = index // cols
        col_index = index % cols
        canvas_widget.grid(row=row_index, column=col_index, padx=10, pady=10)
        canvas.draw()

    # Plot the final accuracies
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(accuracies)), accuracies, color='skyblue')
    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels([f"{i}" for i in scenario_names])
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Accuracies of Simulations')

    # Display accuracy values on top of the bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{accuracy:.3f}', ha='center', va='bottom', fontsize=8)

    # Embed the accuracy plot in the inner frame
    canvas = FigureCanvasTkAgg(fig, master=plots_inner_frame)
    canvas_widget = canvas.get_tk_widget()
    # Calculate row and column index for grid placement
    row_index = rows
    col_index = 0
    canvas_widget.grid(row=row_index, columnspan=cols, padx=10, pady=10)
    canvas.draw()

    # Update the inner frame's width to fit the plots
    plots_inner_frame.update_idletasks()
    inner_width = plots_inner_frame.winfo_width()
    plots_canvas.config(scrollregion=(0, 0, inner_width, plots_inner_frame.winfo_height()))

    # Bind scrollbar to the canvas
    plots_canvas.bind('<Configure>', lambda e: plots_canvas.configure(scrollregion=plots_canvas.bbox("all")))

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
root.geometry("1200x600")

# Main container frame
container_frame = ttk.Frame(root)
container_frame.pack(fill='both', expand=True, pady=10, padx=10)

# Left frame for controls and canvas
left_frame = ttk.Frame(container_frame)
left_frame.pack(side='left', fill='both', expand=True)

# Right frame for the explanation
right_frame = ttk.Frame(container_frame, width=200)
right_frame.pack(side='right', fill='y', padx=10)

# Explanation text
explanation_text = """Welcome to PoisonPlayground!
This tool simulates federated learning scenarios, allowing you to configure various parameters and 
attacks for educational and research purposes. Use the controls on the left to set up your scenarios 
and start the simulation. Results will be displayed graphically."""

explanation_label = ttk.Label(right_frame, text=explanation_text, wraplength=180, justify="left")
explanation_label.pack(pady=10, padx=10)

num_scenarios_var = tk.StringVar(value = 0)
num_trials_var = tk.DoubleVar(value = 1)
seed_var = tk.DoubleVar(value = 20)


# Create an input frame for scenario and trial inputs
input_frame = ttk.Frame(left_frame)
input_frame.pack(pady=10, padx=10, fill='x')

# Place "Number of Scenarios" and "Number of Trials" side by side
num_scenarios_label = ttk.Label(input_frame, text="Number of Scenarios:")
num_scenarios_label.grid(row=0, column=0, padx=5, sticky='w')
num_scenarios_entry = ttk.Entry(input_frame, textvariable=num_scenarios_var)
num_scenarios_entry.grid(row=0, column=1, padx=5, sticky='ew')

num_trials_label = ttk.Label(input_frame, text="Number of Trials:")
num_trials_label.grid(row=0, column=2, padx=20, sticky='w')  # Added padding for visual separation
num_trials_entry = ttk.Entry(input_frame, textvariable=num_trials_var)
num_trials_entry.grid(row=0, column=3, padx=5, sticky='ew')


# "Seed" input
seed_label = ttk.Label(input_frame, text="Seed:")
seed_label.grid(row=0, column=4, padx=20, sticky='w')  # Ensure consistent padding for visual separation
seed_entry = ttk.Entry(input_frame, textvariable=seed_var)
seed_entry.grid(row=0, column=5, padx=5, sticky='ew')

# Adjust grid column configuration to allocate space properly
input_frame.grid_columnconfigure(1, weight=1)
input_frame.grid_columnconfigure(3, weight=1)
input_frame.grid_columnconfigure(5, weight=1)  # Ensure the seed entry expands equally

input_frame.grid_columnconfigure(1, weight=1)
input_frame.grid_columnconfigure(3, weight=1)

# Create an actions frame for buttons
actions_frame = ttk.Frame(left_frame)
actions_frame.pack(pady=10, padx=10, fill='x')

# Place "Create Scenarios" and "Set Defaults" buttons side by side
create_scenarios_button = ttk.Button(actions_frame, text="Create Scenarios", command=create_scenarios)
create_scenarios_button.grid(row=0, column=0, padx=5, sticky='ew')

set_defaults_button = ttk.Button(actions_frame, text="Set Defaults", command=set_default_values)
set_defaults_button.grid(row=0, column=1, padx=5, sticky='ew')

actions_frame.grid_columnconfigure(0, weight=1)
actions_frame.grid_columnconfigure(1, weight=1)

# Ensure the main frame expands to fill the window
main_frame = ttk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=10)

# Scrollable frame setup
main_frame = ttk.Frame(root)  # Main frame to hold the canvas and scrollbar
main_frame.pack(fill='both', expand=True, pady=10)

scenarios_canvas = tk.Canvas(left_frame)
scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=scenarios_canvas.yview)
scenarios_canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
scenarios_canvas.pack(side="left", fill="both", expand=True)

scenarios_frame = ttk.Frame(scenarios_canvas)  # This is the frame we will add widgets to
scenarios_canvas.create_window((0, 0), window=scenarios_frame, anchor="nw")

# This ensures that the canvas frame resizes to fit the inner frame
scenarios_canvas.bind('<Configure>', lambda e: scenarios_canvas.configure(scrollregion=scenarios_canvas.bbox("all")))

run_button = ttk.Button(root, text="Run Simulation", command=start_simulation)
run_button.pack(pady=10)

results_text = tk.StringVar()
results_label = ttk.Label(root, textvariable=results_text)
results_label.pack(pady=10)





root.mainloop()