import os
import numpy as np
import torch
import scipy.stats
import csv
from Source.Util.physics import get_M_ll



def get_mass(data):
    """Calculate the mass and filter out corrupted mass values."""
    mass = get_M_ll(data)
    mass = torch.Tensor(mass).unsqueeze(1)
    corrupted_mass = torch.isnan(mass).squeeze(1)
    mass = mass[~corrupted_mass].squeeze()
    return mass.numpy()

def wasserstein_distance(data_A, data_B):
    """Calculate Wasserstein distance between two datasets."""
    mass_A = get_mass(data_A)
    mass_B = get_mass(data_B)
    return scipy.stats.wasserstein_distance(mass_A, mass_B)

def energy_distance(data_A, data_B):
    """Calculate Energy distance between two datasets."""
    mass_A = get_mass(data_A)
    mass_B = get_mass(data_B)
    return scipy.stats.energy_distance(mass_A, mass_B)

def calculate_metrics(model, teacher_model, data_raw, data_mean, data_std, params, out_dir, n_samples=1_000_000, iterations=5, steps=None):
    """Calculate and save Wasserstein and Energy metrics."""
    true_data = data_raw[:n_samples]
    results = []

    steps = [1,2,3,4,5,6,7,8,9,10,16,32]

    wasser_metrics = []
    energy_metrics = []
    for n in range(iterations):
        cfm_data = teacher_model.sample_and_undo(n_samples)
        wasser_metrics.append(wasserstein_distance(cfm_data, true_data))
        energy_metrics.append(energy_distance(cfm_data, true_data))

    results.append(('CFM', np.mean(wasser_metrics), np.std(wasser_metrics), np.mean(energy_metrics), np.std(energy_metrics)))

    for step in steps:
        wasser_metrics = []
        energy_metrics = []

        for n in range(iterations):
            model.steps = step
            cm_data = model.sample_and_undo(n_samples)
            wasser_metrics.append(wasserstein_distance(cm_data, true_data))
            energy_metrics.append(energy_distance(cm_data, true_data))

        results.append((step, np.mean(wasser_metrics), np.std(wasser_metrics), np.mean(energy_metrics), np.std(energy_metrics)))

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    csv_filepath = os.path.join(out_dir, 'metrics.csv')

    # Export results to CSV
    with open(csv_filepath, 'w', newline='') as csvfile:
        fieldnames = ['step', 'wasser', 'wasser_std', 'energy', 'energy_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for step, wasser, wasser_std, energy, energy_std in results:
            writer.writerow({'step': step, 'wasser': wasser, 'wasser_std': wasser_std, 'energy': energy, 'energy_std': energy_std})

    print(f"CSV file has been saved to: {csv_filepath}")
