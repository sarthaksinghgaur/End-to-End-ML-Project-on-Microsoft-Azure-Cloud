import os
import sys
import dataclasses

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifact', "preprocessor.pkl")

class FeatureEngineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['is_suspicious_distance'] = ((X['distance_from_home'] > 50) & (X['distance_from_last_transaction'] > 10)).astype(int)
        X['high_ratio_purchase'] = (X['ratio_to_median_purchase_price'] > 2).astype(int)
        return X

class DataTransformation:
    def __init__(self):
        self.Data_Transformation_Config = DataTransformationConfig()
    
    def get_data_tranformer_object(self):
        try:
            cat_cols = ['repeat_retailer', 
                        'used_chip',
                        'used_pin_number',
                        'online_order']
            
            num_cols = ['distance_from_home',
                        'distance_from_last_transaction',
                        'ratio_to_median_purchase_price']
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("onehot", OneHotEncoder(drop='if_binary', sparse_output=False)),
                    ("scalar", StandardScaler())
                ]
            )
            num_pipeline = Pipeline([
                ("scalar", StandardScaler())
            ])

            transformer = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
            ])

            preprocessor = Pipeline(steps=[
                                ("feature_engineering", FeatureEngineering()),
                                ("preprocessing", transformer)
                            ])

            return preprocessor

            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor = self.get_data_tranformer_object()

            target_column_name = 'fraud'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying feature engineering and preprocessing on training and testing dataframes.")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applying SMOTE oversampling to balance the training data.")
            smote = SMOTE()
            input_feature_train_arr, target_feature_train_df = smote.fit_resample(input_feature_train_arr, target_feature_train_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path = self.Data_Transformation_Config.preprocessor_file_path,
                obj = preprocessor
            )

            return(
                train_arr,
                test_arr,
                self.Data_Transformation_Config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
