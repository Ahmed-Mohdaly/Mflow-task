# MLflow Experiment Tracking: California Housing

## 📌 Project Overview
This project demonstrates the implementation of MLOps best practices using **MLflow** to track, compare, and manage machine learning experiments. The objective is to train a `GradientBoostingRegressor` on the California Housing dataset using various hyperparameters, log the metrics and models, and identify the optimal configuration.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** scikit-learn, pandas, numpy
* **Experiment Tracking:** MLflow

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd mlflow-task



2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt

  ```


3. Execute the training script to run the 3 experiments:
```bash
python train.py

```


4. Launch the MLflow UI to view and compare the results:
```bash
python -m mlflow ui

```


*Open `http://127.0.0.1:5000` in your browser.*

## 📊 Experiment Results & Comparison

| Experiment | Max Depth | Learning Rate | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| **Run 1** | 3 | 0.10 | 0.5422 | 0.3716 | 0.7756 |
| **Run 2** | 5 | 0.05 | **0.5235** | **0.3541** | **0.7908** |
| **Run 3** | 7 | 0.01 | 0.7042 | 0.5255 | 0.6214 |

<img width="1796" height="800" alt="Screenshot 2026-10-09 235942" src="https://github.com/user-attachments/assets/d342b3eb-1c4c-4002-88ca-93e779c5c42e" />
============================================================================================`
<img width="1200" height="603" alt="Screenshot 2026-10-09 235822" src="https://github.com/user-attachments/assets/5106aa7e-5f4d-4cef-a1f7-2e20df56148c" />





## 🏆 Model Selection & Behavior Analysis

**Selected Model: Run 2 (Max Depth: 5, Learning Rate: 0.05)**

**Q1: Why was Run 2 selected as the best model?**

> beacuse Run 2 achieved the lowest Root Mean Squared Error (**RMSE = 0.5235**), which is the primary evaluation metric for this regression task. It successfully found the optimal balance, minimizing the error distance between predicted and actual housing prices, while also achieving the highest R-squared score (0.7908).

**Q2: Why did Run 1 underperform compared to Run 2?**

> With a `max_depth` of only 3, the model lacked the structural complexity required to capture the underlying, This is a classic case of **underfitting**; the tree was too shallow to learn effectively, resulting in a higher RMSE (0.5422).

**Q3: Why did the more complex Run 3 fail to yield the best results?**

> Although Run 3 had a deeper structure (`max_depth=7`), its exceptionally low `learning_rate` (0.01) prevented the model from converging optimally within the default number of boosting stages. Furthermore, higher depth trees are highly susceptible to **overfitting**, where the model memorizes the training data's noise rather than generalizing well to the validation set. Run 2 hit the "sweet spot" between structural depth and learning pace.
