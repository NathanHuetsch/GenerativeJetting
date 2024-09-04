import os
import time
import numpy as np
import csv

def cm_sample_time(model, n_samples, steps):
    """Measure the time taken to sample `n_samples` using `steps` steps."""
    t_start = time.time()
    model.sample_n(n_samples, steps)
    t_stop = time.time()
    return t_stop - t_start

def cfm_sample_time(teacher_model, n_samples):
    """Measure the time taken to sample `n_samples` using the teacher model."""
    t_start = time.time()
    teacher_model.sample_n(n_samples)
    t_stop = time.time()
    return t_stop - t_start

def calculate_sample_times(model, teacher_model, out_dir, n_samples=1_000_000, iterations=5, steps=15):
    """Calculate and save sampling times for both CM and CFM models."""
    results = []

    # Measure CFM sampling times
    cfm_sample_times = [cfm_sample_time(teacher_model, n_samples) for _ in range(iterations)]
    mean_sample_time = np.mean(cfm_sample_times)
    std_sample_time = np.std(cfm_sample_times)
    results.append(("CFM", mean_sample_time, std_sample_time))

    # Measure CM sampling times for each step
    for step in range(steps):
        cm_sample_times = [cm_sample_time(model, n_samples, step + 1) for _ in range(iterations)]
        mean_sample_time = np.mean(cm_sample_times)
        std_sample_time = np.std(cm_sample_times)
        results.append((step + 1, mean_sample_time, std_sample_time))

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    csv_filepath = os.path.join(out_dir, 'sample_times.csv')

    # Export results to CSV
    with open(csv_filepath, 'w', newline='') as csvfile:
        fieldnames = ['Step', 'Mean_Sample_Time', 'Std_Deviation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step, mean, std in results:
            writer.writerow({'Step': step, 'Mean_Sample_Time': mean, 'Std_Deviation': std})

    print(f"CSV file has been saved to: {csv_filepath}")
