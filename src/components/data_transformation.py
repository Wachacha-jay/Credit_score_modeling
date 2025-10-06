"""
Data Transformation Component
Handles preprocessing, feature engineering, and transformation
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils.logger import logger
from src.utils.exception import DataTransformationException
from src.utils.common import save_object
from src.config.configuration import DataTransformationConfig


class DataTransformation:
    """
    Class for data transformation and preprocessing
    """
    
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize DataTransformation component
        
        Args:
            config: DataTransformationConfig object
        """
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        logger.info("DataTransformation component initialized")
    
    def remove_outliers_iqr(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Remove outliers using IQR method by clipping
        
        Args:
            df: Input DataFrame
            columns: List of columns to process
            
        Returns:
            DataFrame with outliers clipped
        """
        try:
            logger.info("Removing outliers using IQR method...")
            df_copy = df.copy()
            
            for column in columns:
                if column in df_copy.columns:
                    Q1 = df_copy[column].quantile(0.25)
                    Q3 = df_copy[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - self.config.outlier_multiplier * IQR
                    upper = Q3 + self.config.outlier_multiplier * IQR
                    
                    # Clip values
                    original_min = df_copy[column].min()
                    original_max = df_copy[column].max()
                    df_copy[column] = df_copy[column].clip(lower, upper)
                    
                    logger.info(
                        f"{column}: Clipped [{original_min:.2f}, {original_max:.2f}] "
                        f"to [{lower:.2f}, {upper:.2f}]"
                    )
            
            logger.info("Outlier removal completed")
            return df_copy
            
        except Exception as e:
            raise DataTransformationException(e, sys)
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        try:
            logger.info("Encoding categorical features...")
            df_copy = df.copy()
            
            for column in self.config.categorical_features:
                if column in df_copy.columns:
                    le = LabelEncoder()
                    df_copy[column] = le.fit_transform(df_copy[column].astype(str))
                    self.label_encoders[column] = le
                    logger.info(f"Encoded '{column}' with classes: {le.classes_}")
            
            # Encode target column if present
            if self.config.target_column in df_copy.columns:
                le = LabelEncoder()
                df_copy[self.config.target_column] = le.fit_transform(
                    df_copy[self.config.target_column].astype(str)
                )
                self.label_encoders[self.config.target_column] = le
                logger.info(
                    f"Encoded target '{self.config.target_column}' "
                    f"with classes: {le.classes_}"
                )
            
            logger.info("Categorical encoding completed")
            return df_copy
            
        except Exception as e:
            raise DataTransformationException(e, sys)
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple:
        """
        Scale numerical features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        try:
            logger.info("Scaling features...")
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("Feature scaling completed")
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            raise DataTransformationException(e, sys)
    
    def transform_data(self, train_path: str, test_path: str) -> tuple:
        """
        Transform both training and testing data
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to testing data CSV
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        try:
            logger.info("=" * 50)
            logger.info("Data Transformation Started")
            logger.info("=" * 50)
            
            # Load data
            logger.info(f"Loading training data from: {train_path}")
            train_df = pd.read_csv(train_path)
            logger.info(f"Loading testing data from: {test_path}")
            test_df = pd.read_csv(test_path)
            
            # Drop unnecessary columns
            logger.info(f"Dropping columns: {self.config.drop_columns}")
            for col in self.config.drop_columns:
                if col in train_df.columns:
                    train_df = train_df.drop(columns=[col])
                if col in test_df.columns:
                    test_df = test_df.drop(columns=[col])
            
            # Remove outliers
            numerical_cols = self.config.numerical_features
            train_df = self.remove_outliers_iqr(train_df, numerical_cols)
            test_df = self.remove_outliers_iqr(test_df, numerical_cols)
            
            # Encode categorical features
            train_df = self.encode_categorical_features(train_df)
            test_df = self.encode_categorical_features(test_df)
            
            # Separate features and target
            logger.info("Separating features and target...")
            target_col = self.config.target_column
            
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(
                X_train.values, X_test.values
            )
            
            # Save preprocessing objects
            logger.info("Saving preprocessing objects...")
            save_object(self.config.scaler_path, self.scaler)
            save_object(self.config.encoder_path, self.label_encoders)
            
            # Save preprocessor info
            preprocessor_info = {
                'feature_names': X_train.columns.tolist(),
                'numerical_features': self.config.numerical_features,
                'categorical_features': self.config.categorical_features,
                'target_column': self.config.target_column
            }
            save_object(self.config.preprocessor_path, preprocessor_info)
            
            logger.info("=" * 50)
            logger.info("Data Transformation Completed Successfully")
            logger.info("=" * 50)
            
            return (
                X_train_scaled,
                X_test_scaled,
                y_train.values,
                y_test.values,
                X_train.columns.tolist()
            )
            
        except Exception as e:
            raise DataTransformationException(e, sys)