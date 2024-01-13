import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.parameters import Parameters
from Federated_Learning.unpoisoned import normal_run_simulation
from Federated_Learning.labelFlipping import label_flipping_run_simulation
from Federated_Learning.modelPoisoning import model_poisoning_run_simulation

params = Parameters()
selectedAttacks = params.selectedAttacks
selectedDefenses = params.selectedDefenses

# Run the selected attacks and defenses
normal_run_simulation()

if "label_flipping" in selectedAttacks:
    label_flipping_run_simulation()
if "model_poisoning" in selectedAttacks:
    model_poisoning_run_simulation()

if "label_flipping" in selectedDefenses:
    # defense
    pass
if "model_poisoning" in selectedDefenses:
    # defense
    pass