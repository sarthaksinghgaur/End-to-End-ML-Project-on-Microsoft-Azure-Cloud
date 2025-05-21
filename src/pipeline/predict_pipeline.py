import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
        self,
        distance_from_home: float,
        distance_from_last_transaction: float,
        ratio_to_median_purchase_price: float,
        repeat_retailer: int,
        used_chip: int,
        used_pin_number: int,
        online_order: int
    ):
        self.distance_from_home = distance_from_home
        self.distance_from_last_transaction = distance_from_last_transaction
        self.ratio_to_median_purchase_price = ratio_to_median_purchase_price
        self.repeat_retailer = repeat_retailer
        self.used_chip = used_chip
        self.used_pin_number = used_pin_number
        self.online_order = online_order

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "distance_from_home": [self.distance_from_home],
                "distance_from_last_transaction": [self.distance_from_last_transaction],
                "ratio_to_median_purchase_price": [self.ratio_to_median_purchase_price],
                "repeat_retailer": [self.repeat_retailer],
                "used_chip": [self.used_chip],
                "used_pin_number": [self.used_pin_number],
                "online_order": [self.online_order],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)