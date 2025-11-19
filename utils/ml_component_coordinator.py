#!/usr/bin/env python3
"""
ML Component Coordinator for GoalDiggers Platform

Manages ML component initialization order, dependencies, and communication
to optimize startup time from 53.84s to <20s target through:
- Parallel initialization for independent components
- Async communication patterns
- Optimized component loading sequence
- Dependency management and resolution
"""

import asyncio
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import safe symbols for Windows compatibility
try:
    from utils.safe_symbols import get_safe_logger
    logger = get_safe_logger(__name__)
except ImportError:
    # Fallback to regular logger if safe_symbols not available
    pass

class MLComponentCoordinator:
    """
    Coordinates ML component initialization and communication for optimal performance.
    
    Key Features:
    - Parallel initialization for independent components
    - Dependency-aware loading sequence
    - Async communication patterns
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        """Initialize the ML component coordinator."""
        self.start_time = time.time()
        self.components = weakref.WeakValueDictionary()
        self.component_status = {}

        # Load configuration
        try:
            from utils.unified_config_manager import get_unified_config
            self.config = get_unified_config()

            # Get performance configuration
            self.component_timeout = self.config.get_config('performance.component_timeout', 5.0)
            self.startup_timeout = self.config.get_config('performance.startup_timeout', 20.0)

        except Exception as e:
            logger.warning(f"Failed to load unified config: {e}")
            self.config = None
            self.component_timeout = 5.0
            self.startup_timeout = 20.0

        # Initialize dependency resolver
        try:
            from utils.dependency_resolver import get_dependency_resolver
            self.dependency_resolver = get_dependency_resolver()
            logger.info("🔗 Dependency resolver integrated")
        except Exception as e:
            logger.warning(f"Failed to load dependency resolver: {e}")
            self.dependency_resolver = None

        # Initialize enhanced components registry
        self.enhanced_components = {
            'Interface': None,
            'enhanced_team_data_manager': None,
            'enhanced_prediction_display': None
        }
        self.enhanced_component_status = {
            'Interface': 'not_loaded',
            'enhanced_team_data_manager': 'not_loaded',
            'enhanced_prediction_display': 'not_loaded'
        }

        self.initialization_order = [
            'enhanced_prediction_engine',  # Core component - load first
            'adaptive_ensemble',           # Independent - can load in parallel
            'dynamic_trainer'              # Independent - can load in parallel
        ]

        # Component dependencies mapping
        self.dependencies = {
            'enhanced_prediction_engine': [],  # No dependencies
            'adaptive_ensemble': [],           # No dependencies
            'dynamic_trainer': []              # No dependencies
        }

        # Performance tracking
        self.load_times = {}
        self.initialization_complete = False

        logger.info("🎯 ML Component Coordinator initialized with unified config")
    
    async def initialize_components(self) -> Dict[str, Any]:
        """
        Initialize ML components (alias for initialize_components_optimized for compatibility).

        Returns:
            Dictionary containing initialization results and performance metrics
        """
        return await self.initialize_components_optimized()

    async def initialize_components_optimized(self) -> Dict[str, Any]:
        """
        Initialize ML components with optimized parallel loading.

        Returns:
            Dictionary containing initialization results and performance metrics
        """
        logger.info("🚀 Starting optimized ML component initialization...")
        init_start = time.time()

        try:
            # Phase 1: Initialize core component (enhanced_prediction_engine)
            core_result = await self._initialize_core_component()

            # Phase 2: Initialize independent components in parallel
            parallel_results = await self._initialize_parallel_components()

            # Phase 3: Initialize enhanced components (optional)
            enhanced_results = await self._initialize_enhanced_components()

            # Phase 4: Validate all components and establish communication
            validation_result = await self._validate_and_connect_components()

            total_time = time.time() - init_start
            
            # Compile results
            results = {
                'success': True,
                'total_time': total_time,
                'core_component': core_result,
                'parallel_components': parallel_results,
                'enhanced_components': enhanced_results,
                'validation': validation_result,
                'components_loaded': len(self.components),
                'enhanced_components_loaded': len([c for c in self.enhanced_components.values() if c is not None]),
                'performance_target_met': total_time < 15.0,  # 15s target for ML components
                'timestamp': datetime.now().isoformat()
            }
            
            self.initialization_complete = True
            logger.info(f"✅ ML component initialization completed in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ ML component initialization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - init_start,
                'components_loaded': len(self.components)
            }
    
    async def _initialize_core_component(self) -> Dict[str, Any]:
        """Initialize the core enhanced prediction engine."""
        logger.info("🎯 Initializing core component: enhanced_prediction_engine")
        start_time = time.time()
        
        try:
            # Load enhanced prediction engine with timeout
            component = await asyncio.wait_for(
                self._load_component_async('enhanced_prediction_engine'),
                timeout=8.0
            )
            
            load_time = time.time() - start_time
            self.load_times['enhanced_prediction_engine'] = load_time
            
            if component:
                self.components['enhanced_prediction_engine'] = component
                self.component_status['enhanced_prediction_engine'] = 'loaded'
                logger.info(f"✅ Enhanced prediction engine loaded in {load_time:.2f}s")
                return {'success': True, 'load_time': load_time, 'component': 'enhanced_prediction_engine'}
            else:
                logger.warning("⚠️ Enhanced prediction engine returned None, using fallback")
                return {'success': False, 'fallback': True, 'load_time': load_time}
                
        except asyncio.TimeoutError:
            logger.warning("⏰ Enhanced prediction engine loading timed out, using fallback")
            return {'success': False, 'timeout': True, 'load_time': time.time() - start_time}
        except Exception as e:
            logger.error(f"❌ Enhanced prediction engine loading failed: {e}")
            return {'success': False, 'error': str(e), 'load_time': time.time() - start_time}
    
    async def _initialize_parallel_components(self) -> Dict[str, Any]:
        """Initialize independent components in parallel."""
        logger.info("⚡ Initializing parallel components: adaptive_ensemble, dynamic_trainer")
        
        parallel_components = ['adaptive_ensemble', 'dynamic_trainer']
        
        # Create tasks for parallel execution
        tasks = []
        for component_name in parallel_components:
            task = asyncio.create_task(
                self._load_component_with_timeout(component_name, timeout=5.0)
            )
            tasks.append((component_name, task))
        
        # Wait for all parallel tasks to complete
        results = {}
        for component_name, task in tasks:
            try:
                result = await task
                results[component_name] = result
                
                if result['success'] and result.get('component'):
                    self.components[component_name] = result['component']
                    self.component_status[component_name] = 'loaded'
                else:
                    self.component_status[component_name] = 'fallback'
                    
            except Exception as e:
                logger.error(f"❌ Parallel loading failed for {component_name}: {e}")
                results[component_name] = {'success': False, 'error': str(e)}
                self.component_status[component_name] = 'error'
        
        # Calculate parallel loading performance
        total_parallel_time = max(
            result.get('load_time', 0) for result in results.values()
        )
        
        logger.info(f"⚡ Parallel component loading completed in {total_parallel_time:.2f}s")
        
        return {
            'results': results,
            'parallel_time': total_parallel_time,
            'components_loaded': sum(1 for r in results.values() if r.get('success', False))
        }
    
    async def _load_component_with_timeout(self, component_name: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Load a component with timeout protection."""
        start_time = time.time()
        
        try:
            component = await asyncio.wait_for(
                self._load_component_async(component_name),
                timeout=timeout
            )
            
            load_time = time.time() - start_time
            self.load_times[component_name] = load_time
            
            if component:
                logger.info(f"✅ {component_name} loaded in {load_time:.2f}s")
                return {
                    'success': True,
                    'component': component,
                    'load_time': load_time,
                    'component_name': component_name
                }
            else:
                logger.warning(f"⚠️ {component_name} returned None")
                return {
                    'success': False,
                    'fallback': True,
                    'load_time': load_time,
                    'component_name': component_name
                }
                
        except asyncio.TimeoutError:
            load_time = time.time() - start_time
            logger.warning(f"⏰ {component_name} loading timed out after {timeout}s")
            return {
                'success': False,
                'timeout': True,
                'load_time': load_time,
                'component_name': component_name
            }
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"❌ {component_name} loading failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'load_time': load_time,
                'component_name': component_name
            }
    
    async def _load_component_async(self, component_name: str) -> Optional[Any]:
        """Asynchronously load a specific ML component."""
        try:
            if component_name == 'enhanced_prediction_engine':
                from enhanced_prediction_engine import \
                    get_enhanced_prediction_engine
                return await asyncio.to_thread(get_enhanced_prediction_engine)
            
            elif component_name == 'adaptive_ensemble':
                from models.ensemble.adaptive_voting import \
                    get_adaptive_ensemble
                return await asyncio.to_thread(get_adaptive_ensemble)
            
            elif component_name == 'dynamic_trainer':
                from models.realtime.dynamic_trainer import get_dynamic_trainer
                return await asyncio.to_thread(get_dynamic_trainer)
            
            else:
                logger.error(f"Unknown component: {component_name}")
                return None
                
        except Exception as e:
            logger.error(f"Component loading error for {component_name}: {e}")
            return None
    
    async def _validate_and_connect_components(self) -> Dict[str, Any]:
        """Validate loaded components and establish communication patterns."""
        logger.info("🔗 Validating components and establishing communication...")
        
        validation_start = time.time()
        validation_results = {}
        
        # Validate each component
        for component_name, component in self.components.items():
            try:
                if component_name == 'enhanced_prediction_engine':
                    is_valid = hasattr(component, 'predict_match_outcome')
                elif component_name == 'adaptive_ensemble':
                    is_valid = hasattr(component, 'predict')
                elif component_name == 'dynamic_trainer':
                    is_valid = hasattr(component, 'train_model') or hasattr(component, '_train_model')
                else:
                    is_valid = component is not None
                
                validation_results[component_name] = {
                    'valid': is_valid,
                    'status': self.component_status.get(component_name, 'unknown')
                }
                
            except Exception as e:
                logger.error(f"Validation failed for {component_name}: {e}")
                validation_results[component_name] = {
                    'valid': False,
                    'error': str(e)
                }
        
        validation_time = time.time() - validation_start
        
        # Calculate overall validation success
        valid_components = sum(1 for result in validation_results.values() if result.get('valid', False))
        total_components = len(validation_results)
        validation_success_rate = (valid_components / total_components) * 100 if total_components > 0 else 0
        
        logger.info(f"🔗 Component validation completed: {valid_components}/{total_components} valid ({validation_success_rate:.1f}%)")
        
        return {
            'validation_time': validation_time,
            'results': validation_results,
            'valid_components': valid_components,
            'total_components': total_components,
            'success_rate': validation_success_rate
        }
    
    async def _initialize_enhanced_components(self) -> Dict[str, Any]:
        """Initialize enhanced components (design system, team data, prediction display)."""
        logger.info("🎨 Initializing enhanced components...")
        start_time = time.time()

        enhanced_results = {}

        try:
            # Initialize unified design system
            try:
                from dashboard.components.unified_design_system import \
                    get_unified_design_system
                design_system = get_unified_design_system()
                self.enhanced_components['Interface'] = design_system
                self.enhanced_component_status['Interface'] = 'loaded'
                enhanced_results['Interface'] = {'success': True, 'colors_count': len(design_system.brand_colors)}
                logger.info("✅ Unified design system initialized")
            except Exception as e:
                logger.warning(f"⚠️ Unified design system initialization failed: {e}")
                enhanced_results['Interface'] = {'success': False, 'error': str(e)}

            # Initialize enhanced team data manager
            try:
                from utils.enhanced_team_data_manager import \
                    get_enhanced_team_data_manager
                team_manager = get_enhanced_team_data_manager()
                self.enhanced_components['enhanced_team_data_manager'] = team_manager
                self.enhanced_component_status['enhanced_team_data_manager'] = 'loaded'
                enhanced_results['enhanced_team_data_manager'] = {
                    'success': True,
                    'leagues_supported': len(team_manager.get_all_supported_leagues())
                }
                logger.info("✅ Enhanced team data manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced team data manager initialization failed: {e}")
                enhanced_results['enhanced_team_data_manager'] = {'success': False, 'error': str(e)}

            # Initialize enhanced prediction display
            try:
                from dashboard.components.enhanced_prediction_display import \
                    get_enhanced_prediction_display
                design_system = self.enhanced_components.get('Interface')
                prediction_display = get_enhanced_prediction_display(design_system)
                self.enhanced_components['enhanced_prediction_display'] = prediction_display
                self.enhanced_component_status['enhanced_prediction_display'] = 'loaded'
                enhanced_results['enhanced_prediction_display'] = {'success': True, 'animation_duration': prediction_display.animation_duration}
                logger.info("✅ Enhanced prediction display initialized")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced prediction display initialization failed: {e}")
                enhanced_results['enhanced_prediction_display'] = {'success': False, 'error': str(e)}

            load_time = time.time() - start_time
            success_count = sum(1 for result in enhanced_results.values() if result.get('success', False))

            return {
                'success': success_count > 0,  # Success if at least one component loaded
                'load_time': load_time,
                'components_loaded': success_count,
                'total_components': len(enhanced_results),
                'results': enhanced_results
            }

        except Exception as e:
            logger.error(f"Enhanced components initialization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'load_time': time.time() - start_time,
                'components_loaded': 0
            }

    def get_component(self, component_name: str) -> Optional[Any]:
        """Get a loaded component by name."""
        return self.components.get(component_name)

    def get_enhanced_component(self, component_name: str) -> Optional[Any]:
        """Get a loaded enhanced component by name."""
        return self.enhanced_components.get(component_name)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        total_time = time.time() - self.start_time
        
        return {
            'total_coordinator_time': total_time,
            'individual_load_times': self.load_times.copy(),
            'components_loaded': len(self.components),
            'component_status': self.component_status.copy(),
            'initialization_complete': self.initialization_complete,
            'performance_target_met': total_time < 15.0,
            'timestamp': datetime.now().isoformat()
        }

# Global coordinator instance
_coordinator_instance = None

def get_ml_component_coordinator() -> MLComponentCoordinator:
    """Get global ML component coordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = MLComponentCoordinator()
    return _coordinator_instance

async def initialize_ml_components_optimized() -> Dict[str, Any]:
    """Initialize ML components with optimized coordination."""
    coordinator = get_ml_component_coordinator()
    return await coordinator.initialize_components_optimized()

def get_ml_component(component_name: str) -> Optional[Any]:
    """Get a loaded ML component by name."""
    coordinator = get_ml_component_coordinator()
    return coordinator.get_component(component_name)
