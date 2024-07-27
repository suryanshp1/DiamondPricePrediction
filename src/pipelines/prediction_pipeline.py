import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        carat: float,
        depth: float,
        table: float,
        x: float,
        y: float,
        z: float,
        cut: str,
        color: str,
        clarity: str,
    ) -> None:
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df

        except Exception as e:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(e, sys)
