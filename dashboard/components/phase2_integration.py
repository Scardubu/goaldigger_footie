#!/usr/bin/env python3
"""
Phase 2 Integration Component for GoalDiggers Dashboard
Handles display and integration of Phase 2 ML enhancements including ensemble predictions,
confidence scoring, and model agreement indicators.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

from dashboard.components.unified_design_system import \
    get_unified_design_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2IntegrationComponent:
    """Component for integrating Phase 2 ML enhancements into the dashboard."""
    
    def __init__(self):
        """Initialize Phase 2 integration component."""
        self.phase2_enabled = False
        self.performance_monitoring = True
        self.feature_flags = {
            'ensemble_display': True,
            'confidence_tracking': True,
            'model_agreement': True,
            'performance_alerts': True
        }
        
        logger.info("Phase 2 Integration Component initialized")
    
    def render_ensemble_predictions(self, prediction_result: Dict[str, Any]) -> None:
        """Render enhanced ensemble predictions with Phase 2 features using UnifiedDesignSystem."""
        try:
            uds = get_unified_design_system()
            uds.inject_unified_css('integrated')
            if not prediction_result:
                st.warning("No prediction data available")
                return
            # Check if Phase 2 features are available
            is_phase2 = prediction_result.get('metadata', {}).get('phase2_enabled', False)
            if is_phase2:
                self._render_phase2_predictions(prediction_result, uds)
            else:
                self._render_standard_predictions(prediction_result, uds)
        except Exception as e:
            logger.error(f"Failed to render ensemble predictions: {e}")
            st.error("Error displaying predictions")
    
    def _render_phase2_predictions(self, result: Dict[str, Any], uds) -> None:
        """Render Phase 2 enhanced predictions with ensemble details using UnifiedDesignSystem."""
        try:
            predictions = result.get('predictions', {})
            confidence = result.get('confidence', {})
            ensemble_details = result.get('ensemble_details', {})
            uds.create_unified_header("🤖 AI Ensemble Predictions (Phase 2 Enhanced)")
            def card_content():
                st.markdown("<div style='text-align:center;font-size:1.1em;'>AI-powered ensemble probabilities for each outcome</div>", unsafe_allow_html=True)
            uds.create_unified_card(card_content)
            metrics = {
                "🏠 Home Win": f"{predictions.get('home_win', 0)*100:.1f}%",
                "🤝 Draw": f"{predictions.get('draw', 0)*100:.1f}%",
                "✈️ Away Win": f"{predictions.get('away_win', 0)*100:.1f}%"
            }
            uds.create_unified_metric_row(metrics)
            # Ensemble confidence and agreement
            if ensemble_details:
                st.markdown("---")
                self._render_ensemble_confidence(confidence, ensemble_details, uds)
            # Model agreement indicator
            if 'ensemble_agreement' in confidence:
                self._render_model_agreement(confidence['ensemble_agreement'], uds)
            # Performance metrics
            if 'metadata' in result:
                self._render_performance_metrics(result['metadata'], uds)
        except Exception as e:
            logger.error(f"Failed to render Phase 2 predictions: {e}")
            st.error("Error displaying Phase 2 predictions")
    
    def _render_standard_predictions(self, result: Dict[str, Any], uds) -> None:
        """Render standard Phase 1 predictions using UnifiedDesignSystem."""
        try:
            predictions = result.get('predictions', {})
            confidence = result.get('confidence', {})
            uds.create_unified_header("📊 Match Predictions")
            def card_content():
                st.markdown("<div style='text-align:center;font-size:1.1em;'>Model probabilities for each outcome</div>", unsafe_allow_html=True)
            uds.create_unified_card(card_content)
            metrics = {
                "🏠 Home Win": f"{predictions.get('home_win', 0)*100:.1f}%",
                "🤝 Draw": f"{predictions.get('draw', 0)*100:.1f}%",
                "✈️ Away Win": f"{predictions.get('away_win', 0)*100:.1f}%"
            }
            uds.create_unified_metric_row(metrics)
            # Overall confidence
            if 'overall' in confidence:
                st.markdown("---")
                overall_conf = confidence['overall'] * 100
                st.metric(
                    "Overall Confidence", 
                    f"{overall_conf:.1f}%",
                    help="Model confidence in the prediction"
                )
        except Exception as e:
            logger.error(f"Failed to render standard predictions: {e}")
            st.error("Error displaying predictions")
    
    # _render_prediction_card is now obsolete with UnifiedDesignSystem metric row
    
    def _render_ensemble_confidence(self, confidence: Dict[str, Any], ensemble_details: Dict[str, Any], uds) -> None:
        """Render ensemble confidence and model details using UnifiedDesignSystem."""
        try:
            uds.create_unified_header("🎯 Ensemble Intelligence")
            metrics = {
                "Ensemble Confidence": f"{ensemble_details.get('ensemble_confidence', 0.5)*100:.1f}%",
                "Model Agreement": f"{ensemble_details.get('model_agreement', 0.5)*100:.1f}%"
            }
            uds.create_unified_metric_row(metrics)
            # Individual model confidences and method
            individual_confs = ensemble_details.get('individual_confidences', {})
            if individual_confs:
                def card_content():
                    st.markdown("**Individual Model Confidences:**")
                    for model_name, conf in individual_confs.items():
                        conf_pct = conf * 100 if isinstance(conf, (int, float)) else 50
                        st.write(f"• {model_name.title()}: {conf_pct:.1f}%")
                uds.create_unified_card(card_content)
            method = ensemble_details.get('method', 'standard')
            st.caption(f"Ensemble Method: {method.title()}")
        except Exception as e:
            logger.error(f"Failed to render ensemble confidence: {e}")
    
    def _render_model_agreement(self, agreement_score: float, uds) -> None:
        """Render model agreement indicator using UnifiedDesignSystem."""
        try:
            agreement_pct = agreement_score * 100
            if agreement_pct >= 80:
                level = "High"
                color = "#10b981"
                icon = "🟢"
            elif agreement_pct >= 60:
                level = "Medium"
                color = "#f59e0b"
                icon = "🟡"
            else:
                level = "Low"
                color = "#ef4444"
                icon = "🔴"
            def card_content():
                st.markdown(f"<strong style='color: {color};'>{icon} Model Agreement: {level}</strong>", unsafe_allow_html=True)
                st.caption(f"{agreement_pct:.1f}% of models agree on this prediction")
            uds.create_unified_card(card_content)
        except Exception as e:
            logger.error(f"Failed to render model agreement: {e}")
    
    def _render_performance_metrics(self, metadata: Dict[str, Any], uds) -> None:
        """Render performance metrics for Phase 2 using UnifiedDesignSystem."""
        try:
            if not self.performance_monitoring:
                return
            uds.create_unified_header("⚡ Performance Metrics")
            metrics = {
                "Inference Time": f"{metadata.get('inference_time', 0):.3f}s",
                "Model Version": str(metadata.get('model_version', '2.0')),
                "Features Used": str(metadata.get('feature_count', 0))
            }
            uds.create_unified_metric_row(metrics)
            if metadata.get('inference_time', 0) > 1.0:
                st.warning(f"⚠️ Inference time ({metadata.get('inference_time', 0):.3f}s) exceeds 1 second threshold")
        except Exception as e:
            logger.error(f"Failed to render performance metrics: {e}")
    
    def _get_confidence_color(self, probability: float) -> str:
        """Get color based on prediction confidence."""
        if probability >= 60:
            return "#10b981"  # Green for high confidence
        elif probability >= 40:
            return "#f59e0b"  # Orange for medium confidence
        else:
            return "#6b7280"  # Gray for low confidence
    
    def render_phase2_status(self, prediction_engine) -> None:
        """Render Phase 2 status information."""
        try:
            if hasattr(prediction_engine, 'get_phase2_status'):
                status = prediction_engine.get_phase2_status()
                
                with st.expander("🔧 Phase 2 System Status", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Phase 2 Features:**")
                        st.write(f"• Enabled: {'✅' if status['phase2_enabled'] else '❌'}")
                        st.write(f"• Adaptive Ensemble: {'✅' if status['adaptive_ensemble_available'] else '❌'}")
                        st.write(f"• Dynamic Trainer: {'✅' if status['dynamic_trainer_available'] else '❌'}")
                    
                    with col2:
                        st.write("**Performance Thresholds:**")
                        thresholds = status.get('performance_thresholds', {})
                        st.write(f"• Max Inference Time: {thresholds.get('max_inference_time', 1.0)}s")
                        st.write(f"• Max Memory: {thresholds.get('max_memory_mb', 150)}MB")
                        st.write(f"• Min Accuracy: {thresholds.get('min_accuracy', 0.82)*100:.0f}%")
                        
        except Exception as e:
            logger.error(f"Failed to render Phase 2 status: {e}")


def create_phase2_integration() -> Phase2IntegrationComponent:
    """Factory function to create Phase 2 integration component."""
    return Phase2IntegrationComponent()
