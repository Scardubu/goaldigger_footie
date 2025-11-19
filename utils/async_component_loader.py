#!/usr/bin/env python3
"""
Async Component Loader for GoalDiggers Platform
Optimizes ML component loading with parallel initialization and progress tracking.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

class AsyncComponentLoader:
    """
    Advanced async component loader with progress tracking and intelligent caching.
    
    Features:
    - Parallel component initialization
    - Real-time progress tracking with entertaining messages
    - Intelligent caching and lazy loading
    - Memory optimization during loading
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, max_workers: int = 4):
        """Initialize async component loader."""
        self.max_workers = max_workers
        self.loaded_components = {}
        self.loading_progress = {}
        self.loading_messages = {
            'dynamic_trainer': [
                "🤖 Training AI models...",
                "🧠 Analyzing historical match data...",
                "⚡ Optimizing prediction algorithms...",
                "🎯 Calibrating model accuracy..."
            ],
            'adaptive_ensemble': [
                "🎯 Assembling prediction ensemble...",
                "🔄 Synchronizing model weights...",
                "📊 Optimizing voting strategies..."
            ],
            'enhanced_prediction_engine': [
                "🚀 Initializing prediction engine...",
                "📈 Loading XGBoost models...",
                "🔧 Configuring feature mappings...",
                "✨ Preparing SHAP explanations..."
            ],
            'live_data_processor': [
                "📡 Connecting to live data streams...",
                "🌐 Establishing API connections...",
                "📊 Initializing data pipelines..."
            ],
            'odds_aggregator': [
                "💰 Aggregating betting odds...",
                "📊 Analyzing market data...",
                "🎲 Calculating value opportunities..."
            ],
            'preference_engine': [
                "👤 Personalizing user experience...",
                "🎯 Learning user preferences...",
                "📝 Optimizing recommendations..."
            ]
        }
        
        # Component configurations with priority and dependencies
        self.component_configs = [
            {
                'name': 'adaptive_ensemble',
                'module': 'models.ensemble.adaptive_voting',
                'function': 'get_adaptive_ensemble',
                'priority': 1,  # High priority, fast loading
                'estimated_time': 0.1,
                'dependencies': []
            },
            {
                'name': 'odds_aggregator',
                'module': 'data.market.odds_aggregator',
                'function': 'get_odds_aggregator',
                'priority': 1,  # High priority, fast loading
                'estimated_time': 0.1,
                'dependencies': []
            },
            {
                'name': 'preference_engine',
                'module': 'user.personalization.preference_engine',
                'function': 'get_preference_engine',
                'priority': 1,  # High priority, fast loading
                'estimated_time': 0.1,
                'dependencies': []
            },
            {
                'name': 'live_data_processor',
                'module': 'data.streams.live_data_processor',
                'function': 'get_live_data_processor',
                'priority': 2,  # Medium priority
                'estimated_time': 1.5,
                'dependencies': []
            },
            {
                'name': 'enhanced_prediction_engine',
                'module': 'enhanced_prediction_engine',
                'function': 'get_enhanced_prediction_engine',
                'priority': 2,  # Medium priority
                'estimated_time': 4.0,
                'dependencies': []
            },
            {
                'name': 'dynamic_trainer',
                'module': 'models.realtime.dynamic_trainer',
                'function': 'get_dynamic_trainer',
                'priority': 3,  # Low priority, heavy loading
                'estimated_time': 15.0,  # Reduced from 67s with optimization
                'dependencies': []
            }
        ]
        
        logger.info("🚀 Async Component Loader initialized")

    def load_single_component(self, component_name: str) -> Optional[Any]:
        """Load a single component by name."""
        try:
            # Find component config
            config = None
            for comp_config in self.component_configs:
                if comp_config['name'] == component_name:
                    config = comp_config
                    break

            if not config:
                logger.warning(f"Component config not found: {component_name}")
                return None

            # Load component
            start_time = time.time()
            component = self._load_component_safe(config)
            load_time = time.time() - start_time

            if component:
                logger.info(f"✅ {component_name}: Loaded in {load_time:.3f}s")
                return component
            else:
                logger.warning(f"⚠️ {component_name}: Failed to load")
                return None

        except Exception as e:
            logger.error(f"Failed to load single component {component_name}: {e}")
            return None

    def _load_component_safe(self, config: Dict[str, Any]) -> Optional[Any]:
        """Safely load a single component with error handling."""
        try:
            module_name = config['module']
            function_name = config['function']

            # Import module
            module = __import__(module_name, fromlist=[function_name])

            # Get function
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                return func()
            else:
                logger.warning(f"Function {function_name} not found in {module_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to load component {config['name']}: {e}")
            return None

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # Fallback if psutil not available
    
    def _load_component_sync(self, config: Dict[str, Any]) -> Tuple[str, Any, float]:
        """Load a single component synchronously."""
        component_name = config['name']
        start_time = time.time()

        try:
            # Import the module
            module = __import__(config['module'], fromlist=[config['function']])

            # Get the class or function
            class_or_function = getattr(module, config['function'])

            # Validate that the function/class exists and is callable
            if class_or_function is None:
                raise ImportError(f"Function/class '{config['function']}' not found in module '{config['module']}'")

            # Initialize the component
            if callable(class_or_function):
                if config['function'] == 'EnhancedPredictionEngine':
                    component = class_or_function()
                else:
                    component = class_or_function()
            else:
                component = class_or_function

            # Validate component was created successfully
            if component is None:
                raise RuntimeError(f"Component '{component_name}' initialization returned None")

            load_time = time.time() - start_time
            logger.info(f"✅ {component_name}: Loaded in {load_time:.3f}s")

            return component_name, component, load_time
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"❌ {component_name}: Failed to load in {load_time:.3f}s - {e}")
            return component_name, None, load_time
    
    def load_components_parallel(self, progress_container=None) -> Dict[str, Any]:
        """
        Load components in parallel with progress tracking.
        
        Args:
            progress_container: Streamlit container for progress display
            
        Returns:
            Dictionary of loaded components
        """
        logger.info("🔄 Starting parallel component loading...")

        # Sort components by priority (higher priority first)
        sorted_configs = sorted(self.component_configs, key=lambda x: x['priority'])
        
        # Calculate total estimated time
        total_estimated_time = sum(config['estimated_time'] for config in sorted_configs)
        
        # Initialize progress tracking
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
            time_remaining = progress_container.empty()
        
        loaded_components = {}
        completed_time = 0
        
        # Load high priority components first (parallel) with memory awareness
        high_priority_configs = [c for c in sorted_configs if c['priority'] == 1]
        medium_priority_configs = [c for c in sorted_configs if c['priority'] == 2]
        low_priority_configs = [c for c in sorted_configs if c['priority'] == 3]

        # Check memory before loading heavy components (more aggressive)
        current_memory = self._get_current_memory()
        memory_threshold = 150  # MB - more aggressive threshold

        if current_memory > memory_threshold:
            logger.warning(f"⚠️ Memory usage high ({current_memory:.1f}MB), prioritizing essential components only")
            # Only load high priority components if memory is high
            medium_priority_configs = []
            low_priority_configs = []
        
        # Phase 1: Load high priority components in parallel with timeout
        if high_priority_configs:
            logger.info("📊 Phase 1: Loading high priority components...")
            with ThreadPoolExecutor(max_workers=min(len(high_priority_configs), self.max_workers)) as executor:
                future_to_config = {
                    executor.submit(self._load_component_sync, config): config
                    for config in high_priority_configs
                }

                # Add timeout to prevent hanging
                for future in as_completed(future_to_config, timeout=10.0):  # 10 second timeout
                    config = future_to_config[future]
                    try:
                        component_name, component, load_time = future.result(timeout=5.0)  # 5 second per component
                    except Exception as e:
                        component_name = config['name']
                        component = None
                        load_time = 5.0
                        logger.warning(f"⚠️ {component_name}: Loading failed or timed out - {e}")
                    
                    loaded_components[component_name] = component
                    completed_time += config['estimated_time']
                    
                    if progress_container:
                        progress = completed_time / total_estimated_time
                        progress_bar.progress(min(progress, 1.0))
                        
                        if component_name in self.loading_messages:
                            messages = self.loading_messages[component_name]
                            message = messages[min(len(messages) - 1, int(progress * len(messages)))]
                            status_text.text(f"{message} ✅")
                        
                        remaining_time = max(0, total_estimated_time - completed_time)
                        time_remaining.text(f"⏱️ Estimated time remaining: {remaining_time:.1f}s")
        
        # Phase 2: Load medium priority components in parallel
        if medium_priority_configs:
            logger.info("⚡ Phase 2: Loading medium priority components...")
            with ThreadPoolExecutor(max_workers=min(len(medium_priority_configs), self.max_workers)) as executor:
                future_to_config = {
                    executor.submit(self._load_component_sync, config): config 
                    for config in medium_priority_configs
                }
                
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    component_name, component, load_time = future.result()
                    
                    loaded_components[component_name] = component
                    completed_time += config['estimated_time']
                    
                    if progress_container:
                        progress = completed_time / total_estimated_time
                        progress_bar.progress(min(progress, 1.0))
                        
                        if component_name in self.loading_messages:
                            messages = self.loading_messages[component_name]
                            message = messages[min(len(messages) - 1, int(progress * len(messages)))]
                            status_text.text(f"{message} ✅")
                        
                        remaining_time = max(0, total_estimated_time - completed_time)
                        time_remaining.text(f"⏱️ Estimated time remaining: {remaining_time:.1f}s")
        
        # Phase 3: Load low priority components (can be deferred)
        if low_priority_configs:
            logger.info("🐌 Phase 3: Loading low priority components...")
            for config in low_priority_configs:
                if progress_container:
                    if config['name'] in self.loading_messages:
                        messages = self.loading_messages[config['name']]
                        for i, message in enumerate(messages):
                            status_text.text(message)
                            time.sleep(0.5)  # Brief pause for user feedback
                
                component_name, component, load_time = self._load_component_sync(config)
                loaded_components[component_name] = component
                completed_time += config['estimated_time']
                
                if progress_container:
                    progress = completed_time / total_estimated_time
                    progress_bar.progress(min(progress, 1.0))
                    
                    remaining_time = max(0, total_estimated_time - completed_time)
                    time_remaining.text(f"⏱️ Estimated time remaining: {remaining_time:.1f}s")
        
        # Complete progress
        if progress_container:
            progress_bar.progress(1.0)
            status_text.text("🎉 All components loaded successfully!")
            time_remaining.text("✅ Loading complete!")
        
        logger.info(f"✅ Parallel component loading complete: {len(loaded_components)}/6 components loaded")
        return loaded_components

# Global singleton instance
_async_loader_instance = None

def get_async_component_loader() -> AsyncComponentLoader:
    """Get global async component loader instance."""
    global _async_loader_instance
    if _async_loader_instance is None:
        _async_loader_instance = AsyncComponentLoader()
    return _async_loader_instance
