import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_cofig=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors":KNeighborsRegressor(),
                "catboosting":CatBoostRegressor(verbose=False),
                "Adaboost":AdaBoostRegressor()
            }
            params={
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.05],
                    "max_depth": [3, 5, 10]
                },
                "Linear Regression": {
                    # No hyperparameters for basic Linear Regression, can use Ridge/Lasso if needed
                },
                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7, 10],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },
                "CatBoost": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [3, 5, 7]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,param=params)
            
            # to get model model score from dict
            best_model_score=max(sorted(model_report.values()))
            # to get best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on the training and testing dataset")

            save_object(
                file_path=self.model_trainer_cofig.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_squre=r2_score(y_test,predicted)
        
        except Exception as e:
            raise CustomException(e,sys)
