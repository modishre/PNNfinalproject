from preprocess import preprocess_data
from early_exit import EarlyExitMechanismNDIF
from metrics import estimate_efficiency
from energy_tracking import CarbonTracker

def main():
    import os
    os.makedirs("results/logs", exist_ok=True)

    tracker = CarbonTracker()
    tracker.start()

    try:
        # Load data
        data = preprocess_data("ag_news")
        print(f"Sample data: {data[0]}")

        # Initialize model with NDIF
        model = EarlyExitMechanismNDIF(confidence_threshold=0.8)
        results = []

        # Run early exit experiments
        for idx, sample in enumerate(data):
            try:
                result = model.run_with_early_exit(sample["text"])
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {idx}: {repr(e)}")
                continue

        # Estimate efficiency
        metrics = estimate_efficiency(results, total_layers=32)
        print(metrics)

    except Exception as e:
        print(f"Error in main processing: {repr(e)}")

    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
