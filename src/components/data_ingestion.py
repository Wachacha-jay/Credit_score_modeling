"""
Data Ingestion Component
Handles loading and splitting of raw data
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils.logger import logger
from src.utils.exception import DataIngestionException
from src.config.configuration import DataIngestionConfig


class DataIngestion:
    """
    Class for ingesting and splitting data
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion component
        
        Args:
            config: DataIngestionConfig object
        """
        self.config = config
        logger.info("DataIngestion component initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        
        Returns:
            DataFrame with raw data
            
        Raises:
            DataIngestionException: If loading fails
        """
        try:
            logger.info(f"Loading data from: {self.config.raw_data_path}")
            
            if not os.path.exists(self.config.raw_data_path):
                raise FileNotFoundError(
                    f"Data file not found: {self.config.raw_data_path}"
                )
            
            df = pd.read_csv(self.config.raw_data_path)
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            raise DataIngestionException(e, sys)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
            
        Raises:
            DataIngestionException: If validation fails
        """
        try:
            logger.info("Validating data...")
            
            # Check for empty dataframe
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            # Check for required columns (basic check)
            required_columns = ['loan_status']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
            
            logger.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            raise DataIngestionException(e, sys)
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            DataIngestionException: If splitting fails
        """
        try:
            logger.info("Splitting data into train and test sets...")
            
            stratify_column = None
            if self.config.stratify and 'loan_status' in df.columns:
                stratify_column = df['loan_status']
                logger.info("Using stratified split on 'loan_status'")
            
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_column
            )
            
            logger.info(f"Train set shape: {train_df.shape}")
            logger.info(f"Test set shape: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            raise DataIngestionException(e, sys)
    
    def save_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save train and test data to CSV files
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            
        Raises:
            DataIngestionException: If saving fails
        """
        try:
            logger.info("Saving train and test data...")
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)
            
            # Save data
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Train data saved at: {self.config.train_data_path}")
            logger.info(f"Test data saved at: {self.config.test_data_path}")
            
        except Exception as e:
            raise DataIngestionException(e, sys)
    
    def initiate_data_ingestion(self) -> tuple:
        """
        Main method to execute data ingestion pipeline
        
        Returns:
            Tuple of (train_data_path, test_data_path)
            
        Raises:
            DataIngestionException: If any step fails
        """
        try:
            logger.info("=" * 50)
            logger.info("Data Ingestion Started")
            logger.info("=" * 50)
            
            # Load data
            df = self.load_data()
            
            # Validate data
            self.validate_data(df)
            
            # Split data
            train_df, test_df = self.split_data(df)
            
            # Save data
            self.save_data(train_df, test_df)
            
            logger.info("=" * 50)
            logger.info("Data Ingestion Completed Successfully")
            logger.info("=" * 50)
            
            return (
                str(self.config.train_data_path),
                str(self.config.test_data_path)
            )
            
        except Exception as e:
            raise DataIngestionException(e, sys)