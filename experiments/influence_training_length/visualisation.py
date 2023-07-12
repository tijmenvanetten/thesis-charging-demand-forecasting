import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

import sys 
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../src'))

from evaluation.utils import load_scores_from_json

def plot(train_lengths, mae_local, mae_global):

    # Plotting
    sns.set(style='whitegrid')

    # MAE plot
    plt.subplot(2, 1, 1)
    plt.title("MAE of NHiTS (Global) and NHiTS (Local) with varying training size")
    plt.plot(train_lengths, mae_local, marker='o', label='NHiTS (Local)')
    plt.plot(train_lengths, mae_global, marker='o', label='NHiTS (Global)')
    plt.xlabel('Training Length')
    plt.ylabel('MAE')
    plt.legend()

    # Delta plot
    plt.subplot(2, 1, 2)
    plt.title("Δ MAE of NHiTS (Global) and NHiTS (Local)")
    plt.plot(train_lengths, mae_local - mae_global, marker='o', c='r', label='Δ')
    plt.xlabel('Training Length')
    plt.ylabel('MAE')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show the figure
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experimental Python script.")
    parser.add_argument('--file', type=str, default='./results/boulder.json', help='Path to results.json file.')
    args = parser.parse_args()

    # Load scores
    scores = load_scores_from_json(args.file)
    train_length = np.array(scores['train_length'])[::-1]
    mae_local = np.array(scores['NHiTS (Local)'])[::-1]
    mae_global = np.array(scores['NHiTS (Global)'])[::-1]

    # Plot
    plot(train_length, mae_local, mae_global)

