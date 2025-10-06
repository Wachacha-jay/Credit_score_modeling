"""
Custom exception handler for the loan approval system
"""
import sys
from src.utils.logger import logger


class LoanApprovalException(Exception):
    """
    Custom exception class for loan approval system
    """
    
    def __init__(self, error_message: Exception, error_detail: sys):
        """
        Initialize custom exception
        
        Args:
            error_message: The error message
            error_detail: System error details
        """
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )
        
    @staticmethod
    def get_detailed_error_message(error: Exception, error_detail: sys) -> str:
        """
        Generate detailed error message with file name and line number
        
        Args:
            error: The exception
            error_detail: System error details
            
        Returns:
            Detailed error message string
        """
        _, _, exc_tb = error_detail.exc_info()
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            
            error_message = (
                f"Error occurred in script: [{file_name}] "
                f"at line number: [{line_number}] "
                f"error message: [{str(error)}]"
            )
        else:
            error_message = f"Error: {str(error)}"
            
        return error_message
    
    def __str__(self):
        """String representation of the exception"""
        return self.error_message


class DataIngestionException(LoanApprovalException):
    """Exception raised during data ingestion"""
    pass


class DataTransformationException(LoanApprovalException):
    """Exception raised during data transformation"""
    pass


class ModelTrainingException(LoanApprovalException):
    """Exception raised during model training"""
    pass


class ModelEvaluationException(LoanApprovalException):
    """Exception raised during model evaluation"""
    pass


class PredictionException(LoanApprovalException):
    """Exception raised during prediction"""
    pass