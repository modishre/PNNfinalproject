def estimate_efficiency(results, total_layers):
    if not results:
        print("No results available to estimate efficiency. Ensure the processing pipeline is functioning correctly.")
        return {"avg_layers_used": 0, "skipped_layers": total_layers}

    avg_layers_used = sum(r.get("layer_used", 0) for r in results) / len(results)
    skipped_layers = total_layers - avg_layers_used
    print(f"Average Layers Used: {avg_layers_used:.2f}, Layers Skipped: {skipped_layers:.2f}")
    return {"avg_layers_used": avg_layers_used, "skipped_layers": skipped_layers}
