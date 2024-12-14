#PNN Final Project: Early Exits and Environmental Impact

##Overview

This project demonstrates the implementation of early exit mechanisms in transformer-based models to optimize energy efficiency while maintaining task performance. Using GPT-2 as a substitute for LLaMA 2 (7B), the code evaluates the trade-offs between performance, efficiency, and environmental impact by dynamically halting inference at intermediate layers based on confidence thresholds.

The project uses the IMDb dataset for binary text classification and tracks energy consumption and emissions using CodeCarbon.


<img width="561" alt="image" src="https://github.com/user-attachments/assets/f79e325e-4283-42e9-ba06-6c4bd6fa9d77" />


Setup and Installation


Clone the repository:
pip install -r requirements.txt

How to Run

Execute the main script:
python main.py

The script will:
Load the IMDb dataset for binary classification.
Apply early exit mechanisms during model inference.
Track energy consumption using CodeCarbon.
Save results in the results/ folder, including a plot of early exit layers and confidence scores (results_plot.png).
Code Components

1. main.py
Handles the overall project pipeline:
Loads the IMDb dataset.
Initializes the EarlyExitRunner for early exits.
Tracks energy usage using CodeCarbon.
Saves visualization of results.
2. early_exit.py
Implements the EarlyExitRunner class.
Logic for monitoring prediction confidence at each layer.
Halts inference when confidence exceeds the predefined threshold (default: 0.8).
3. flops_calculator.py
Calculates the computational FLOPs for layers utilized during inference.
4. codecarbon_logging.py
Tracks energy consumption and emissions using the CodeCarbon library.
5. dataset_handler.py
Preprocesses datasets and ensures compatibility with the model.
Supports IMDb dataset and AG News dataset (future extension).
