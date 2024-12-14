import os
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from datasets import load_dataset
from early_exit import EarlyExitRunner


def save_results_plot(results, output_dir):
    """
    Generate plots from the results and save them.
    """
    layers = [res['layer'] for res in results]
    confidences = [res['confidence'] for res in results]
    sample_indices = list(range(len(results)))

    plt.figure(figsize=(10, 6))

    # Plot layer exit
    plt.subplot(2, 1, 1)
    plt.plot(sample_indices, layers, marker="o", linestyle="-")
    plt.title("Early Exit Layer per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Layer")

    # Plot confidence scores
    plt.subplot(2, 1, 2)
    plt.plot(sample_indices, confidences, marker="o", linestyle="-", color="orange")
    plt.title("Confidence per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Confidence")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_plot.png"))
    plt.close()


def main():
    # Load dataset
    datasets = load_dataset("imdb")
    test_data = datasets["test"]

    # Set up the emissions tracker
    output_dir = "./simplemodel/results"
    os.makedirs(output_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=output_dir, measure_power_secs=1)
    tracker.start()

    # Initialize the EarlyExitRunner
    model_name = "gpt2"  # Replace with your desired model name
    runner = EarlyExitRunner(model_name=model_name)
    runner.set_threshold(0.8)  # Adjust the threshold if needed

    # Process the dataset and collect results
    results = runner.run_with_early_exit(test_data)

    # Save plots
    save_results_plot(results, output_dir)
    print(f"Plots saved in {output_dir}")

    tracker.stop()
    print("Processing complete.")


if __name__ == "__main__":
    main()
