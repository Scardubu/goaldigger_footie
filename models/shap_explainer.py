#!/usr/bin/env python3
"""
SHAP Explainer System for GoalDiggers Platform

Provides SHAP-based explanations for prediction interpretability
with interactive visualizations and feature importance analysis.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Try to import SHAP with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    SHAP-based model explainer for prediction interpretability.
    """
    
    def __init__(self):
        """Initialize the SHAP explainer system."""
        self.explainers = {}
        self.feature_names = []
        self.background_data = None
        self.explanation_cache = {}
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explanations will be simulated")
        
        logger.info("SHAP explainer system initialized")
    
    def initialize_explainer(self, 
                           model: Any,
                           background_data: pd.DataFrame,
                           model_type: str = "tree",
                           feature_names: List[str] = None) -> bool:
        """
        Initialize SHAP explainer for a model.
        
        Args:
            model: Trained model to explain
            background_data: Background dataset for SHAP
            model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
            feature_names: Names of features
            
        Returns:
            True if successful
        """
        try:
            if not SHAP_AVAILABLE:
                # Create mock explainer for demonstration
                self.explainers[model_type] = self._create_mock_explainer(background_data)
                self.feature_names = feature_names or [f"feature_{i}" for i in range(len(background_data.columns))]
                self.background_data = background_data
                logger.info(f"Mock SHAP explainer created for {model_type}")
                return True
            
            # Create appropriate SHAP explainer
            if model_type == "tree":
                explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, background_data)
            elif model_type == "deep":
                explainer = shap.DeepExplainer(model, background_data)
            elif model_type == "kernel":
                explainer = shap.KernelExplainer(model.predict, background_data)
            else:
                # Default to kernel explainer
                explainer = shap.KernelExplainer(model.predict, background_data)
            
            self.explainers[model_type] = explainer
            self.feature_names = feature_names or [f"feature_{i}" for i in range(len(background_data.columns))]
            self.background_data = background_data
            
            logger.info(f"SHAP explainer initialized for {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            return False
    
    def _create_mock_explainer(self, background_data: pd.DataFrame) -> Dict[str, Any]:
        """Create a mock explainer for demonstration purposes."""
        return {
            'type': 'mock',
            'background_data': background_data,
            'feature_count': len(background_data.columns)
        }
    
    def explain_prediction(self, 
                         instance: Union[pd.DataFrame, np.ndarray],
                         model_type: str = "tree",
                         return_expected_value: bool = True) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a prediction instance.
        
        Args:
            instance: Instance to explain
            model_type: Type of model explainer to use
            return_expected_value: Whether to return expected value
            
        Returns:
            Dictionary with SHAP values and metadata
        """
        try:
            if model_type not in self.explainers:
                logger.error(f"No explainer found for model type: {model_type}")
                return self._create_mock_explanation(instance)
            
            explainer = self.explainers[model_type]
            
            if not SHAP_AVAILABLE or explainer.get('type') == 'mock':
                return self._create_mock_explanation(instance)
            
            # Generate SHAP values
            if isinstance(instance, pd.DataFrame):
                shap_values = explainer.shap_values(instance.values)
            else:
                shap_values = explainer.shap_values(instance)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            explanation = {
                'shap_values': shap_values,
                'expected_value': explainer.expected_value if return_expected_value else None,
                'feature_names': self.feature_names,
                'instance_values': instance.values[0] if isinstance(instance, pd.DataFrame) else instance,
                'prediction_value': np.sum(shap_values) + (explainer.expected_value if return_expected_value else 0)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}")
            return self._create_mock_explanation(instance)
    
    def _create_mock_explanation(self, instance: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Create a mock explanation for demonstration."""
        if isinstance(instance, pd.DataFrame):
            feature_count = len(instance.columns)
            instance_values = instance.values[0]
        else:
            feature_count = len(instance) if instance.ndim == 1 else instance.shape[1]
            instance_values = instance if instance.ndim == 1 else instance[0]
        
        # Generate realistic mock SHAP values
        np.random.seed(42)  # For reproducible mock data
        shap_values = np.random.normal(0, 0.1, feature_count)
        
        # Make some features more important
        important_features = np.random.choice(feature_count, size=min(5, feature_count), replace=False)
        shap_values[important_features] *= 3
        
        return {
            'shap_values': shap_values,
            'expected_value': 0.5,
            'feature_names': self.feature_names[:feature_count] if self.feature_names else [f"feature_{i}" for i in range(feature_count)],
            'instance_values': instance_values,
            'prediction_value': 0.5 + np.sum(shap_values),
            'is_mock': True
        }
    
    def render_waterfall_plot(self, explanation: Dict[str, Any], title: str = "SHAP Waterfall Plot") -> None:
        """
        Render a waterfall plot showing feature contributions.
        
        Args:
            explanation: SHAP explanation dictionary
            title: Plot title
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        expected_value = explanation.get('expected_value', 0)
        
        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        
        # Take top 10 features for clarity
        top_indices = sorted_indices[:10]
        top_shap_values = shap_values[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]
        
        # Create waterfall data
        cumulative_values = [expected_value]
        for shap_val in top_shap_values:
            cumulative_values.append(cumulative_values[-1] + shap_val)
        
        # Create the plot
        fig = go.Figure()
        
        # Add bars for each feature
        colors = ['red' if val < 0 else 'green' for val in top_shap_values]
        
        fig.add_trace(go.Waterfall(
            name="SHAP Values",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(top_shap_values) + ["total"],
            x=["Expected Value"] + top_feature_names + ["Prediction"],
            textposition="outside",
            text=[f"{expected_value:.3f}"] + [f"{val:+.3f}" for val in top_shap_values] + [f"{cumulative_values[-1]:.3f}"],
            y=[expected_value] + list(top_shap_values) + [cumulative_values[-1]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=500,
            xaxis_title="Features",
            yaxis_title="Prediction Value"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_force_plot(self, explanation: Dict[str, Any], title: str = "SHAP Force Plot") -> None:
        """
        Render a force plot showing positive and negative contributions.
        
        Args:
            explanation: SHAP explanation dictionary
            title: Plot title
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        instance_values = explanation['instance_values']
        expected_value = explanation.get('expected_value', 0)
        
        # Separate positive and negative contributions
        positive_mask = shap_values > 0
        negative_mask = shap_values < 0
        
        positive_features = np.array(feature_names)[positive_mask]
        positive_values = shap_values[positive_mask]
        positive_instance_values = instance_values[positive_mask]
        
        negative_features = np.array(feature_names)[negative_mask]
        negative_values = shap_values[negative_mask]
        negative_instance_values = instance_values[negative_mask]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Positive Contributions', 'Negative Contributions'),
            vertical_spacing=0.1
        )
        
        # Positive contributions
        if len(positive_values) > 0:
            sorted_pos_idx = np.argsort(positive_values)[::-1][:10]  # Top 10
            fig.add_trace(
                go.Bar(
                    x=positive_features[sorted_pos_idx],
                    y=positive_values[sorted_pos_idx],
                    name="Positive",
                    marker_color='green',
                    text=[f"Value: {positive_instance_values[i]:.3f}" for i in sorted_pos_idx],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # Negative contributions
        if len(negative_values) > 0:
            sorted_neg_idx = np.argsort(np.abs(negative_values))[::-1][:10]  # Top 10 by magnitude
            fig.add_trace(
                go.Bar(
                    x=negative_features[sorted_neg_idx],
                    y=negative_values[sorted_neg_idx],
                    name="Negative",
                    marker_color='red',
                    text=[f"Value: {negative_instance_values[i]:.3f}" for i in sorted_neg_idx],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Features")
        fig.update_yaxes(title_text="SHAP Value")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_importance_plot(self, 
                                     explanations: List[Dict[str, Any]], 
                                     title: str = "Feature Importance") -> None:
        """
        Render feature importance plot from multiple explanations.
        
        Args:
            explanations: List of SHAP explanations
            title: Plot title
        """
        if not explanations:
            st.warning("No explanations provided for feature importance plot")
            return
        
        # Aggregate SHAP values across all explanations
        all_shap_values = np.array([exp['shap_values'] for exp in explanations])
        feature_names = explanations[0]['feature_names']
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
        
        # Sort by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:15]  # Top 15 features
        
        sorted_importance = mean_abs_shap[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create the plot
        fig = go.Figure(go.Bar(
            x=sorted_importance,
            y=sorted_names,
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_explanation_summary(self, explanation: Dict[str, Any]) -> None:
        """
        Render a summary of the SHAP explanation.
        
        Args:
            explanation: SHAP explanation dictionary
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        instance_values = explanation['instance_values']
        prediction_value = explanation.get('prediction_value', 0)
        expected_value = explanation.get('expected_value', 0)
        
        # Calculate summary statistics
        total_positive = np.sum(shap_values[shap_values > 0])
        total_negative = np.sum(shap_values[shap_values < 0])
        most_important_idx = np.argmax(np.abs(shap_values))
        
        # Create summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Prediction",
                f"{prediction_value:.3f}",
                delta=f"{prediction_value - expected_value:+.3f}" if expected_value else None
            )
        
        with col2:
            st.metric(
                "Positive Impact",
                f"+{total_positive:.3f}",
                delta=f"{len(shap_values[shap_values > 0])} features"
            )
        
        with col3:
            st.metric(
                "Negative Impact",
                f"{total_negative:.3f}",
                delta=f"{len(shap_values[shap_values < 0])} features"
            )
        
        with col4:
            st.metric(
                "Top Feature",
                feature_names[most_important_idx][:15] + "..." if len(feature_names[most_important_idx]) > 15 else feature_names[most_important_idx],
                delta=f"{shap_values[most_important_idx]:+.3f}"
            )
        
        # Show disclaimer for mock data
        if explanation.get('is_mock'):
            st.info("📝 Note: This is simulated SHAP data for demonstration. Install SHAP library for real explanations.")

# Global instance for easy access
shap_explainer = SHAPExplainer()


class ModelEnsemble:
    """
    Model ensemble system with intelligent voting and performance tracking.
    """

    def __init__(self):
        """Initialize the model ensemble system."""
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.voting_strategy = "weighted"  # "simple", "weighted", "stacking"

        logger.info("Model ensemble system initialized")

    def add_model(self,
                  model_id: str,
                  model: Any,
                  weight: float = 1.0,
                  model_type: str = "classifier") -> bool:
        """
        Add a model to the ensemble.

        Args:
            model_id: Unique identifier for the model
            model: Trained model object
            weight: Weight for voting (higher = more influence)
            model_type: Type of model ("classifier", "regressor")

        Returns:
            True if successful
        """
        try:
            self.models[model_id] = {
                'model': model,
                'type': model_type,
                'added_at': pd.Timestamp.now(),
                'prediction_count': 0
            }

            self.model_weights[model_id] = weight
            self.model_performance[model_id] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'prediction_times': [],
                'error_count': 0
            }

            logger.info(f"Added model to ensemble: {model_id} (weight: {weight})")
            return True

        except Exception as e:
            logger.error(f"Failed to add model {model_id}: {e}")
            return False

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the ensemble.

        Args:
            model_id: Model identifier to remove

        Returns:
            True if successful
        """
        try:
            if model_id in self.models:
                del self.models[model_id]
                del self.model_weights[model_id]
                del self.model_performance[model_id]

                logger.info(f"Removed model from ensemble: {model_id}")
                return True
            else:
                logger.warning(f"Model not found in ensemble: {model_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove model {model_id}: {e}")
            return False

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make ensemble predictions using the configured voting strategy.

        Args:
            X: Input features for prediction

        Returns:
            Dictionary with ensemble prediction and individual model results
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        individual_predictions = {}
        individual_probabilities = {}
        prediction_times = {}

        # Get predictions from each model
        for model_id, model_info in self.models.items():
            try:
                start_time = time.time()

                model = model_info['model']

                # Get prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    pred = model.predict(X)
                    individual_probabilities[model_id] = proba
                else:
                    pred = model.predict(X)
                    individual_probabilities[model_id] = None

                individual_predictions[model_id] = pred

                # Track timing
                prediction_time = time.time() - start_time
                prediction_times[model_id] = prediction_time
                self.model_performance[model_id]['prediction_times'].append(prediction_time)

                # Update prediction count
                self.models[model_id]['prediction_count'] += 1

            except Exception as e:
                logger.error(f"Model {model_id} prediction failed: {e}")
                self.model_performance[model_id]['error_count'] += 1
                individual_predictions[model_id] = None
                individual_probabilities[model_id] = None

        # Apply voting strategy
        ensemble_result = self._apply_voting_strategy(
            individual_predictions,
            individual_probabilities
        )

        return {
            'ensemble_prediction': ensemble_result['prediction'],
            'ensemble_probability': ensemble_result.get('probability'),
            'confidence': ensemble_result.get('confidence', 0.0),
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'prediction_times': prediction_times,
            'voting_strategy': self.voting_strategy
        }

    def _apply_voting_strategy(self,
                             predictions: Dict[str, Any],
                             probabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the configured voting strategy."""
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}

        if not valid_predictions:
            raise ValueError("No valid predictions from ensemble models")

        if self.voting_strategy == "simple":
            return self._simple_majority_voting(valid_predictions)
        elif self.voting_strategy == "weighted":
            return self._weighted_voting(valid_predictions, probabilities)
        elif self.voting_strategy == "stacking":
            return self._stacking_voting(valid_predictions, probabilities)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

    def _simple_majority_voting(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Simple majority voting."""
        # Flatten all predictions
        all_preds = []
        for pred in predictions.values():
            if isinstance(pred, np.ndarray):
                all_preds.extend(pred.tolist())
            else:
                all_preds.append(pred)

        # Find most common prediction
        from collections import Counter
        vote_counts = Counter(all_preds)
        ensemble_pred = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[ensemble_pred] / len(all_preds)

        return {
            'prediction': ensemble_pred,
            'confidence': confidence
        }

    def _weighted_voting(self,
                        predictions: Dict[str, Any],
                        probabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted voting based on model weights and performance."""
        if not probabilities or all(p is None for p in probabilities.values()):
            # Fall back to weighted prediction averaging
            weighted_sum = 0
            total_weight = 0

            for model_id, pred in predictions.items():
                weight = self.model_weights.get(model_id, 1.0)
                performance_weight = self.model_performance.get(model_id, {}).get('accuracy', 0.5)
                final_weight = weight * (1 + performance_weight)

                if isinstance(pred, np.ndarray):
                    pred_value = pred[0] if len(pred) > 0 else 0
                else:
                    pred_value = pred

                weighted_sum += pred_value * final_weight
                total_weight += final_weight

            ensemble_pred = weighted_sum / total_weight if total_weight > 0 else 0

            return {
                'prediction': ensemble_pred,
                'confidence': 0.7  # Default confidence
            }

        # Weighted probability averaging
        weighted_proba_sum = None
        total_weight = 0

        for model_id, proba in probabilities.items():
            if proba is not None:
                weight = self.model_weights.get(model_id, 1.0)
                performance_weight = self.model_performance.get(model_id, {}).get('accuracy', 0.5)
                final_weight = weight * (1 + performance_weight)

                if weighted_proba_sum is None:
                    weighted_proba_sum = proba * final_weight
                else:
                    weighted_proba_sum += proba * final_weight

                total_weight += final_weight

        if weighted_proba_sum is not None and total_weight > 0:
            ensemble_proba = weighted_proba_sum / total_weight
            ensemble_pred = np.argmax(ensemble_proba, axis=1)[0] if ensemble_proba.ndim > 1 else np.argmax(ensemble_proba)
            confidence = np.max(ensemble_proba)

            return {
                'prediction': ensemble_pred,
                'probability': ensemble_proba,
                'confidence': confidence
            }

        # Fallback
        return self._simple_majority_voting(predictions)

    def _stacking_voting(self,
                        predictions: Dict[str, Any],
                        probabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Stacking-based ensemble (simplified version)."""
        # For now, use weighted voting as stacking requires a meta-learner
        # In a full implementation, this would train a meta-model
        return self._weighted_voting(predictions, probabilities)

    def update_model_performance(self,
                               model_id: str,
                               accuracy: float,
                               precision: float = None,
                               recall: float = None,
                               f1_score: float = None) -> None:
        """
        Update performance metrics for a model.

        Args:
            model_id: Model identifier
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall
            f1_score: Model F1 score
        """
        if model_id in self.model_performance:
            perf = self.model_performance[model_id]
            perf['accuracy'] = accuracy
            if precision is not None:
                perf['precision'] = precision
            if recall is not None:
                perf['recall'] = recall
            if f1_score is not None:
                perf['f1_score'] = f1_score

            logger.info(f"Updated performance for {model_id}: accuracy={accuracy:.3f}")

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics."""
        stats = {
            'model_count': len(self.models),
            'voting_strategy': self.voting_strategy,
            'total_predictions': sum(
                model_info['prediction_count'] for model_info in self.models.values()
            ),
            'models': {}
        }

        for model_id, model_info in self.models.items():
            perf = self.model_performance[model_id]
            avg_time = np.mean(perf['prediction_times']) if perf['prediction_times'] else 0

            stats['models'][model_id] = {
                'weight': self.model_weights[model_id],
                'prediction_count': model_info['prediction_count'],
                'accuracy': perf['accuracy'],
                'avg_prediction_time': avg_time,
                'error_count': perf['error_count'],
                'error_rate': perf['error_count'] / max(1, model_info['prediction_count'])
            }

        return stats

    def render_ensemble_dashboard(self):
        """Render ensemble performance dashboard."""
        st.markdown("## 🤖 Model Ensemble Dashboard")

        stats = self.get_ensemble_stats()

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Models in Ensemble", stats['model_count'])

        with col2:
            st.metric("Total Predictions", stats['total_predictions'])

        with col3:
            st.metric("Voting Strategy", stats['voting_strategy'].title())

        with col4:
            avg_accuracy = np.mean([
                model_stats['accuracy'] for model_stats in stats['models'].values()
            ]) if stats['models'] else 0
            st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")

        # Model performance comparison
        if stats['models']:
            st.markdown("### 📊 Model Performance Comparison")

            model_df = pd.DataFrame(stats['models']).T
            model_df.index.name = 'Model'

            col1, col2 = st.columns(2)

            with col1:
                # Accuracy comparison
                fig = px.bar(
                    x=model_df.index,
                    y=model_df['accuracy'],
                    title="Model Accuracy Comparison",
                    labels={'x': 'Model', 'y': 'Accuracy'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Prediction time comparison
                fig = px.bar(
                    x=model_df.index,
                    y=model_df['avg_prediction_time'],
                    title="Average Prediction Time",
                    labels={'x': 'Model', 'y': 'Time (seconds)'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed model table
            st.markdown("### 📋 Detailed Model Statistics")
            st.dataframe(model_df, use_container_width=True)

# Global instance for easy access
model_ensemble = ModelEnsemble()
