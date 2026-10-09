import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
california_dataset = fetch_california_housing()




X_features = california_dataset.data
y_target = california_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)


def run_experiment(experiment_name, my_max_depth, my_learning_rate):
    with mlflow.start_run(run_name=experiment_name):
        
        mlflow.log_param("max_depth", my_max_depth)
        mlflow.log_param("learning_rate", my_learning_rate)
        
        model = GradientBoostingRegressor(max_depth=my_max_depth, learning_rate=my_learning_rate, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)


        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAPE", mape)


        
        mlflow.sklearn.log_model(model, "saved_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"])




if __name__ == "__main__":
    run_experiment(experiment_name="Run 1", my_max_depth=3, my_learning_rate=0.1)
    run_experiment(experiment_name="Run 2", my_max_depth=5, my_learning_rate=0.05)
    run_experiment(experiment_name="Run 3", my_max_depth=7, my_learning_rate=0.01)