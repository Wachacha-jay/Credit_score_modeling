"""
Common utility functions
"""
import os
import sys
import yaml
import pickle
import json
from pathlib import Path
from typing import Any, Dict
from box import ConfigBox
from ensure import ensure_annotations

from src.utils.logger import logger
from src.utils.exception import LoanApprovalException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read yaml file and return ConfigBox object
    
    Args:
        path_to_yaml: Path to yaml file
        
    Returns:
        ConfigBox object with yaml content
        
    Raises:
        LoanApprovalException: If file not found or invalid yaml
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        raise LoanApprovalException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create list of directories
    
    Args:
        path_to_directories: List of paths to create
        verbose: Log creation details
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except Exception as e:
        raise LoanApprovalException(e, sys)


@ensure_annotations
def save_object(file_path: Path, obj: Any):
    """
    Save Python object as pickle file
    
    Args:
        file_path: Path to save the object
        obj: Object to save
        
    Raises:
        LoanApprovalException: If saving fails
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise LoanApprovalException(e, sys)


@ensure_annotations
def load_object(file_path: Path) -> Any:
    """
    Load Python object from pickle file
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded Python object
        
    Raises:
        LoanApprovalException: If loading fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            
        logger.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        raise LoanApprovalException(e, sys)


@ensure_annotations
def save_json(path: Path, data: Dict):
    """
    Save dictionary as JSON file
    
    Args:
        path: Path to save JSON
        data: Dictionary to save
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise LoanApprovalException(e, sys)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load JSON file
    
    Args:
        path: Path to JSON file
        
    Returns:
        ConfigBox with JSON content
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON file loaded from: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise LoanApprovalException(e, sys)


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size of file in KB
    
    Args:
        path: Path to file
        
    Returns:
        File size as string
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        raise LoanApprovalException(e, sys)