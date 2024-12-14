# LLAMA7Btesting/visualizations.py
import matplotlib.pyplot as plt

def plot_results(results, output_dir):
    layers_used = [r["layer_used"] for r in results]
    plt.hist(layers_used, bins=10, edgecolor="k")
    plt.xlabel("Layers Used")
    plt.ylabel("Frequency")
    plt.title("Distribution of Layers Used in Early Exit")
    plt.savefig(f"{output_dir}/layers_used_distribution.png")
    plt.show()
