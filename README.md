# Electric Motor Nonlinear Regime Analysis

## Overview

This project explores electric motor behavior using data-driven methods to identify hidden operating regimes without explicitly imposing physical models. The goal was to see if machine learning could discover natural groupings in motor operation that correspond to known physics.

## Key Results

- Identified 3 distinct operating regimes: motoring, regenerative braking, and idle
- Strong nonlinear behavior observed in torque-speed relationship
- PCA shows ~69% variance explained in 2 components
- Gradient Boosting achieved R² ≈ 0.984

## Visualizations

*Torque vs Speed showing three automatically discovered regimes*

![Torque Plot](torque_plot.png)

**Interactive Dashboard:**  
https://Pratikshat22.github.io/Electric-motor-regime-analysis/electric_motor_analysis_complete.html

## Methods Used

- K-Means Clustering
- PCA & t-SNE
- Random Forest & Gradient Boosting
- Residual-based Nonlinearity Detection

## Dataset

Electric motor torque dataset from Kaggle:  
https://www.kaggle.com/datasets/graxlmaxl/identifying-the-physics-behind-an-electric-motor

## Files

- `electric_motor_analysis_complete.html` — interactive dashboard
- `analysis_script.py` — analysis code
- `torque_plot.png` — main visualization

---

*Made by a physics student trying to understand motors without reading textbooks first.*
