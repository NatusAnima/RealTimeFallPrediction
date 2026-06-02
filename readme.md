# Real-Time EEG-based Fall Prediction System

## Overview

This project implements a rigorous machine learning pipeline to evaluate and run an EEG-based fall detection system. It detects unexpected falls in real-time by combining **state-space features extracted from brainwave signals (EEG)** with **physical movement data (Accelerometer)**.

To ensure high accuracy and minimal false alarms in a real-world wearable context, the system uses a **Late Fusion Probabilistic Architecture**. It relies on a lightweight heuristic CPU gate to check for dynamic physical movement before committing heavy computational resources to independent Random Forest models for EEG and IMU. The final fall decision is mapped using a Logistic Regression fusion layer.

## Core Architecture & Strategy

- **Dual-Modality Sensing**: Combines a 3-axis Accelerometer (ACC) and a single-channel EEG (Channel R6).
- **Heuristic CPU Gate**: Acts as a low-power first pass calculating free-fall deviation. If the dynamic heuristic score is below a threshold (meaning gravity is normal), the system safely ignores the event and saves CPU cycles.
- **Independent Feature Extraction**: For events passing the CPU gate, the system extracts **N4SID state-space matrices** from the EEG and **deterministic physical variables** (SMA, Variance, Min Mag, Peak Jerk) from the ACC.
- **Late Fusion Classification**: A logistic regression model (`mdl_fusion`) takes the independent probability scores (`p_eeg` and `p_imu`) from two distinct Random Forests, alongside the `heuristic_score`, to predict the final probabilistic confidence of a fall (`p_fall >= 0.90`).
- **Out-of-Fold Training**: Prevents data leakage by utilizing 5-Fold Cross-Validation to generate unmemorized probabilities for the fusion layer.
- **Multithreading**: The architecture heavily utilizes MATLAB's Parallel Computing Toolbox. Model training is parallelized across subjects (`parfor`), and real-time live predictions are offloaded to background workers (`parfeval`) to ensure UI and DAQ responsiveness.

## System Architecture

![Real-Time Pipeline Architecture](./flowchart.png)

## Main Entry Points

The project has been decoupled into distinct executable scripts based on your use case:

### 1. `run_live_system.m` (Real-Time Live Monitor)

Orchestrates a live connection to a Python Brainflow DAQ (via TCP) and processes continuous real-time data.

- Features a **Real-Time GUI** displaying scrolling plots for the 256ms EEG Window and the ACC Magnitude against the dynamic gate threshold.
- Uses **Asynchronous Multithreading** to push heavy N4SID calculations to a background thread, preventing UI freezing or packet loss.
- Displays a dynamic Color-Coded Status Lamp (Monitoring vs. Fall Detected).

### 2. `train_model.m` (Global Model Generator)

Generates the three core machine learning models required by the live system.

- Processes the entire historical dataset using parallel workers (`parfor`).
- Executes 5-Fold Cross-Validation to generate Out-of-Fold features for the fusion layer.
- Saves the EEG Random Forest, IMU Random Forest, and Fusion Logistic Regression into `trained_model.mat`.

### 3. `run_offline_metrics_gui.m` (Offline Metrics Explorer)

A user-friendly GUI specifically for evaluating the trained model against historical `.edf` data without writing code.

- Features a dropdown to select individual subjects or "All Subjects".
- Displays a progress bar during evaluation.
- Outputs detailed metrics directly to a built-in text console: Sensitivity, Specificity, Balanced Accuracy, and precise False Positive breakdowns (Class 0 expected misclassification vs. Class 2 silence misclassification).

### 4. `master_pipeline.m` (Legacy LOSO Validator)

The original monolithic script used to rigorously validate the algorithm architecture using a strict Leave-One-Subject-Out (LOSO) cross-validation method. It ensures the model can generalize to completely unseen users without data leakage.

## Decoupled Processing Modules

The signal processing logic has been broken down into strict Input/Output modules to support decoupled testing and live execution:

- **`preprocess_signal.m`**: Filters EEG and calculates normalized 3D ACC magnitude.
- **`calculate_imu_heuristic.m`**: Computes continuous deviation from baseline gravity to act as a lightweight gate.
- **`extract_features.m`**: Wraps MATLAB's `n4sid` to extraeadct multi-dimensional state-space feature vectors.
- **`extract_imu_features.m`**: Computes deterministic kinematic variables (SMA, Variance, Min Magnitude, Peak Jerk).
- **`predict_fall.m` / `predict_fall_wrapper.m`**: Standard and background-worker wrappers for executing the full 3-model fusion prediction.
- **`normalized_acc.m`**: Applies Savitzky-Golay filtering to smooth raw accelerometer channels.

## Usage

1. **Setup Workspace**: Ensure your MATLAB workspace is set to the project root directory containing the `Raw_Data/` folder (with `All34_table.mat`, `Label_Table.mat`, and `Filtered/*.edf`).
2. **Train the Model**: Execute `train_model.m` to generate `trained_model.mat`.
3. **Explore Metrics**: Run `run_offline_metrics_gui.m` to explore how the model performs on the historical dataset via the interactive UI.
4. **Go Live**: Start the Python Brainflow DAQ (broadcasting to `127.0.0.1:5005`), then execute `run_live_system.m` to monitor live streams and predictions.
