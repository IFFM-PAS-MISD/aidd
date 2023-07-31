import os
import json


os.chdir('/home/aijjeh/Desktop/Phd_Projects/PC_uint_cell_dispersion_curves/dataset/Keras_tuner_dir/PhC_dispersion_diagrams/trial_09/')
with open('trial.json', 'r') as f:
    hyperparameters = json.load(f)

print(hyperparameters['hyperparameters']['values'])
