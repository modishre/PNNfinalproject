#PNN Final Project: Early Exits and Environmental Impact

##Overview

This project demonstrates the implementation of early exit mechanisms in transformer-based models to optimize energy efficiency while maintaining task performance. Using GPT-2 as a substitute for LLaMA 2 (7B), the code evaluates the trade-offs between performance, efficiency, and environmental impact by dynamically halting inference at intermediate layers based on confidence thresholds.

The project uses the IMDb dataset for binary text classification and tracks energy consumption and emissions using CodeCarbon.

Directory Structure - please refer to this directory
.
├── simplemodel/
│   ├── main.py                # Main script to run the project
│   ├── early_exit.py          # Implementation of early exit mechanisms
│   ├── flops_calculator.py    # Calculates FLOPs for utilized layers
│   ├── codecarbon_logging.py  # Tracks energy usage with CodeCarbon
│   ├── dataset_handler.py     # Handles dataset loading and preprocessing
│   ├── results/
│       └── results_plot.png   # Visualization of results
└── README.md                  # Project documentation

Directory Structure for LLAMA7B - the actual project code I wrote
.
├── LLAMA7Btesting/
│   ├── main.py                # Main script to run the project
│   ├── early_exit.py          # Implementation of early exit mechanisms
│   ├── metrics.py             # Layers output
│   ├── energy_tracking.py     # Tracks energy usage with CodeCarbon
│   ├── preprocess.py          # Handles dataset loading and preprocessing
│   ├── results/
└── README.md                  

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
