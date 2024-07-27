import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass
import os, sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info(
                "Splitting dependent and independent variables from train and test data"
            )

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                # "KNeighbors": KNeighborsRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "LinearRegression": LinearRegression(),
            }

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models
            )
            print("Model Report: \n", model_report)
            print("========================================================\n")
            logging.info("Model Report: \n", model_report)

            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(
                f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}\n"
            )
            print("========================================================\n")
            logging.info(
                f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}\n"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

        except Exception as e:
            raise CustomException(e, sys)
