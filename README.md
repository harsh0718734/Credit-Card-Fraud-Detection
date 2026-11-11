# Credit Card Fraud Detection Pipeline | Python & Machine Learning

## 📌 Project Overview
An advanced, end-to-end Machine Learning classification system engineered in Python to isolate and identify fraudulent credit card activities. Operating on highly imbalanced transaction profiles (284,807 total rows), the model leverages multi-tree ensemble learning, feature evaluation matrices, and robust diagnostic metrics to safeguard financial transaction architectures.

## 🎯 Objectives
- Process and explore an enterprise-grade financial ledger to evaluate structural distribution variances.
- Address severe class imbalances by calculating exact mathematical outlier fractions across transaction arrays.
- Train and fine-tune an optimized Random Forest Classification model with multi-threaded performance setups.
- Run intensive diagnostic audits including Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).
- Provide visual clarity using multi-variable correlation heatmaps and confusion matrix density plots.

## 🛠️ Tools & Technologies
- **Python (VS Code)** — Operational development environment and execution engine
- **Pandas & NumPy** — Descriptive statistical analysis, row parsing, and data matrix manipulation
- **Scikit-Learn** — Model training, train-test splitting workflows, and algorithmic evaluation diagnostics
- **Matplotlib & Seaborn** — Correlation heatmap generation, layout scaling, and exploratory data plotting

## 📊 Pipeline Architecture & Engineering
- **Descriptive Profiling:** Separates fraud segments from valid ledger indices to compute a strict outlier fraction ratio across transaction arrays.
- **Statistical Boundary Tracing:** Calculates explicit financial value limits (`Amount` summaries) across valid and invalid records to map spending skew behaviors.
- **Data Splitting & Multithreading:** Implements a strict `80/20` train-test validation boundary matrix with an optimized `n_jobs=-1` configuration to leverage maximum local processor power during tree fitting.
- **Algorithmic Fitting:** Utilizes an ensemble Random Forest framework to construct multi-tree decision splits, ensuring robust accuracy parameters against data fluctuations.

## 🔍 Model Evaluation Metrics
Because plain Accuracy scores fail on highly imbalanced datasets, the pipeline runs an intensive statistical diagnostic suite to verify target parameters under skewed conditions:
1. **Accuracy Score** — Tracks absolute correct predictions against baseline testing subsets.
2. **Precision & Recall** — Maximizes true positive captures while aggressively shrinking false positive risks.
3. **F1-Score & MCC** — Establishes true statistical prediction trust, balancing precision and recall values into an integrated index.
4. **Confusion Matrix Heatmap** — Maps exact numeric values for True Negatives, False Positives, False Negatives, and True Positives.

## 📁 Project Files
- `fraud_detection.py` — The core Python script handling data processing, model fitting, and evaluation matrices.

## 🚀 How to Run
1. Download the `fraud_detection.py` script into your local workspace folder.
2. Ensure you have the source `creditcard.csv` dataset in the same directory.
3. Fire up your terminal window and ensure your dependencies are installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
4. Run the master pipeline script to execute the model and output your visual charts:
   ```bash
   python fraud_detection.py
   ```
