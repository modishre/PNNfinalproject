from codecarbon import EmissionsTracker


class CarbonTracker:
    def __init__(self, output_dir="results/logs/"):
        self.tracker = EmissionsTracker(output_dir=output_dir)

    def start(self):
        self.tracker.start()

    def stop(self):
        emissions = self.tracker.stop()
        print(f"Total CO₂ emissions: {emissions:.2f} kg")
        return emissions
