import matplotlib.pyplot as plt


def plot_tradeoffs(results, thresholds):
    avg_layers_used = [np.mean(r["layers_used"]) for r in results]
    emissions = [r["emissions"] for r in results]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, avg_layers_used, label="Average Layers Used", marker="o")
    plt.plot(thresholds, emissions, label="Emissions (kg CO₂)", marker="o")
    plt.title("Trade-Off: Layers vs Emissions")
    plt.xlabel("Early Exit Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid()
    plt.savefig("./simplemodel/results/tradeoff_plot.png")
    plt.show()
