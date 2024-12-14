from codecarbon import EmissionsTracker

def track_emissions(func, *args, **kwargs):
    tracker = EmissionsTracker()
    tracker.start()
    result = func(*args, **kwargs)
    emissions = tracker.stop()
    print(f"Estimated emissions: {emissions} kg CO₂")
    return result
