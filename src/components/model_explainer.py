"""
Model Explainer Component
Handles SHAP-based model interpretability
"""
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

from src.utils.logger import logger
from src.utils.exception import LoanApprovalException


class ModelExplainer:
    """
    Class for model interpretability using SHAP
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize ModelExplainer
        
        Args:
            model: Trained model object
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        logger.info("ModelExplainer initialized")
    
    def create_explainer(self, X_background: np.ndarray = None):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background data for explainer (optional)
        """
        try:
            logger.info("Creating SHAP explainer...")
            
            # Use TreeExplainer for tree-based models
            if hasattr(self.model, 'tree_'):
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer")
            # Use Explainer for other models
            else:
                if X_background is not None:
                    self.explainer = shap.Explainer(
                        self.model.predict, 
                        X_background
                    )
                else:
                    self.explainer = shap.Explainer(self.model)
                logger.info("Using standard Explainer")
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def calculate_shap_values(
        self, 
        X: np.ndarray,
        max_samples: int = 100
    ):
        """
        Calculate SHAP values for given data
        
        Args:
            X: Input data
            max_samples: Maximum number of samples to explain
        """
        try:
            if self.explainer is None:
                logger.warning("Explainer not created. Creating now...")
                self.create_explainer()
            
            logger.info("Calculating SHAP values...")
            
            # Limit samples if needed
            if len(X) > max_samples:
                logger.info(f"Using {max_samples} samples for SHAP calculation")
                X = X[:max_samples]
            
            self.shap_values = self.explainer.shap_values(X)
            logger.info("SHAP values calculated successfully")
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values
        
        Returns:
            DataFrame with feature importance
        """
        try:
            if self.shap_values is None:
                raise ValueError("SHAP values not calculated yet")
            
            logger.info("Calculating feature importance...")
            
            # Handle multi-output SHAP values
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1]  # Use positive class
            else:
                shap_vals = self.shap_values
            
            # Calculate mean absolute SHAP values
            importance = np.abs(shap_vals).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            logger.info("Feature importance calculated")
            return importance_df
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def plot_summary(
        self, 
        X: np.ndarray,
        save_path: Optional[str] = None,
        plot_type: str = 'dot'
    ):
        """
        Create SHAP summary plot
        
        Args:
            X: Input data
            save_path: Path to save the plot
            plot_type: Type of plot ('dot', 'bar', 'violin')
        """
        try:
            if self.shap_values is None:
                raise ValueError("SHAP values not calculated yet")
            
            logger.info(f"Creating SHAP summary plot (type: {plot_type})...")
            
            plt.figure(figsize=(10, 8))
            
            # Handle multi-output SHAP values
            shap_vals = self.shap_values
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1]  # Use positive class
            
            shap.summary_plot(
                shap_vals,
                X,
                feature_names=self.feature_names,
                plot_type=plot_type,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Summary plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def plot_waterfall(
        self, 
        X: np.ndarray,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Create waterfall plot for a single prediction
        
        Args:
            X: Input data
            sample_idx: Index of sample to explain
            save_path: Path to save the plot
        """
        try:
            if self.shap_values is None:
                raise ValueError("SHAP values not calculated yet")
            
            logger.info(f"Creating waterfall plot for sample {sample_idx}...")
            
            # Handle multi-output SHAP values
            shap_vals = self.shap_values
            expected_value = self.explainer.expected_value
            
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1]
                expected_value = self.explainer.expected_value[1]
            
            # Create explanation object
            explanation = shap.Explanation(
                values=shap_vals[sample_idx],
                base_values=expected_value,
                data=X[sample_idx],
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Waterfall plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def plot_dependence(
        self,
        X: np.ndarray,
        feature: str,
        save_path: Optional[str] = None
    ):
        """
        Create dependence plot for a feature
        
        Args:
            X: Input data
            feature: Feature name to plot
            save_path: Path to save the plot
        """
        try:
            if self.shap_values is None:
                raise ValueError("SHAP values not calculated yet")
            
            logger.info(f"Creating dependence plot for feature: {feature}...")
            
            # Handle multi-output SHAP values
            shap_vals = self.shap_values
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                shap_vals,
                X,
                feature_names=self.feature_names,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Dependence plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> dict:
        """
        Get detailed explanation for a single prediction
        
        Args:
            X: Input data
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        try:
            if self.shap_values is None:
                raise ValueError("SHAP values not calculated yet")
            
            # Handle multi-output SHAP values
            shap_vals = self.shap_values
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1]
            
            # Get SHAP values for the sample
            sample_shap = shap_vals[sample_idx]
            
            # Create explanation dictionary
            explanation = {
                'feature_contributions': {},
                'top_positive': [],
                'top_negative': []
            }
            
            # Get feature contributions
            for i, feat_name in enumerate(self.feature_names):
                explanation['feature_contributions'][feat_name] = float(sample_shap[i])
            
            # Sort contributions
            sorted_contribs = sorted(
                explanation['feature_contributions'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top positive and negative contributors
            for feat, val in sorted_contribs:
                if val > 0:
                    explanation['top_positive'].append({feat: val})
                else:
                    explanation['top_negative'].append({feat: val})
            
            logger.info(f"Generated explanation for sample {sample_idx}")
            return explanation
            
        except Exception as e:
            raise LoanApprovalException(e, sys)