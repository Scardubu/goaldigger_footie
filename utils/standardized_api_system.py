"""
Standardized API System for GoalDiggers Platform
Provides consistent interfaces for inter-component communication and API standardization.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, Union

from utils.comprehensive_error_handler import (
    APIException,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    GoalDiggersException,
    ValidationException,
)

logger = logging.getLogger(__name__)

class APIResponseStatus(Enum):
    """Standard API response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class APIResponse:
    """Standardized API response structure."""
    status: APIResponseStatus
    data: Any = None
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "errors": self.errors,
            "metadata": self.metadata
        }

class APIInterface(Protocol):
    """Protocol for standardized API interfaces."""

    def get_version(self) -> str:
        """Get API version."""
        ...

    def get_capabilities(self) -> List[str]:
        """Get list of API capabilities."""
        ...

    def validate_request(self, request: Any) -> bool:
        """Validate an incoming request."""
        ...

class ComponentAPI(ABC):
    """Base class for component APIs."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.version = "1.0.0"
        self._capabilities: List[str] = []
        self._middleware: List[Callable] = []

    @abstractmethod
    def get_version(self) -> str:
        """Get component API version."""
        return self.version

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get component capabilities."""
        return self._capabilities.copy()

    def add_capability(self, capability: str):
        """Add a capability to the component."""
        if capability not in self._capabilities:
            self._capabilities.append(capability)

    def add_middleware(self, middleware: Callable):
        """Add middleware to the component."""
        self._middleware.append(middleware)

    def _apply_middleware(self, request: Any) -> Any:
        """Apply middleware to request."""
        for middleware in self._middleware:
            try:
                request = middleware(request)
            except Exception as e:
                logger.warning(f"Middleware failed: {e}")
        return request

class PredictionAPI(ComponentAPI):
    """Standardized API for prediction components."""

    def __init__(self):
        super().__init__("prediction")
        self.add_capability("predict_match")
        self.add_capability("get_confidence")
        self.add_capability("validate_teams")

    def predict_match(self, home_team: str, away_team: str, **kwargs) -> APIResponse:
        """
        Predict match outcome.

        Args:
            home_team: Home team name
            away_team: Away team name
            **kwargs: Additional prediction parameters

        Returns:
            APIResponse with prediction results
        """
        try:
            # Validate input
            if not self._validate_teams(home_team, away_team):
                return APIResponse(
                    status=APIResponseStatus.ERROR,
                    message="Invalid team names",
                    errors=["Home and away teams must be different and valid"]
                )

            # Apply middleware
            request_data = self._apply_middleware({
                "home_team": home_team,
                "away_team": away_team,
                **kwargs
            })

            # Get prediction (to be implemented by subclasses)
            prediction = self._get_prediction(request_data)

            return APIResponse(
                status=APIResponseStatus.SUCCESS,
                data=prediction,
                message="Prediction generated successfully"
            )

        except Exception as e:
            logger.error(f"Prediction API error: {e}")
            return APIResponse(
                status=APIResponseStatus.ERROR,
                message="Prediction failed",
                errors=[str(e)]
            )

    def _validate_teams(self, home_team: str, away_team: str) -> bool:
        """Validate team names."""
        if not home_team or not away_team:
            return False
        if home_team == away_team:
            return False
        if len(home_team.strip()) < 2 or len(away_team.strip()) < 2:
            return False
        return True

    def _get_prediction(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction results (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _get_prediction")

class DatabaseAPI(ComponentAPI):
    """Standardized API for database components."""

    def __init__(self):
        super().__init__("database")
        self.add_capability("connect")
        self.add_capability("query")
        self.add_capability("insert")
        self.add_capability("update")
        self.add_capability("delete")

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        Execute database query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            APIResponse with query results
        """
        try:
            # Apply middleware
            request_data = self._apply_middleware({
                "query": query,
                "params": params or {}
            })

            # Execute query (to be implemented by subclasses)
            results = self._execute_query(request_data)

            return APIResponse(
                status=APIResponseStatus.SUCCESS,
                data=results,
                message="Query executed successfully"
            )

        except Exception as e:
            logger.error(f"Database API error: {e}")
            return APIResponse(
                status=APIResponseStatus.ERROR,
                message="Query execution failed",
                errors=[str(e)]
            )

    def _execute_query(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute query (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _execute_query")

class DataProcessingAPI(ComponentAPI):
    """Standardized API for data processing components."""

    def __init__(self):
        super().__init__("data_processing")
        self.add_capability("process_match_data")
        self.add_capability("validate_data")
        self.add_capability("transform_data")

    def process_data(self, data: Any, operation: str, **kwargs) -> APIResponse:
        """
        Process data with specified operation.

        Args:
            data: Data to process
            operation: Processing operation
            **kwargs: Additional processing parameters

        Returns:
            APIResponse with processed data
        """
        try:
            # Apply middleware
            request_data = self._apply_middleware({
                "data": data,
                "operation": operation,
                **kwargs
            })

            # Process data (to be implemented by subclasses)
            processed_data = self._process_data(request_data)

            return APIResponse(
                status=APIResponseStatus.SUCCESS,
                data=processed_data,
                message=f"Data processed successfully with operation: {operation}"
            )

        except Exception as e:
            logger.error(f"Data processing API error: {e}")
            return APIResponse(
                status=APIResponseStatus.ERROR,
                message="Data processing failed",
                errors=[str(e)]
            )

    def _process_data(self, request_data: Dict[str, Any]) -> Any:
        """Process data (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _process_data")

class APIRegistry:
    """Registry for managing component APIs."""

    def __init__(self):
        self._apis: Dict[str, ComponentAPI] = {}
        self._api_versions: Dict[str, str] = {}

    def register_api(self, name: str, api: ComponentAPI):
        """Register an API with the registry."""
        self._apis[name] = api
        self._api_versions[name] = api.get_version()
        logger.info(f"Registered API: {name} v{api.get_version()}")

    def get_api(self, name: str) -> Optional[ComponentAPI]:
        """Get an API from the registry."""
        return self._apis.get(name)

    def list_apis(self) -> List[str]:
        """List all registered APIs."""
        return list(self._apis.keys())

    def get_api_info(self) -> Dict[str, Any]:
        """Get information about all registered APIs."""
        return {
            "apis": {
                name: {
                    "version": api.get_version(),
                    "capabilities": api.get_capabilities()
                }
                for name, api in self._apis.items()
            },
            "total_apis": len(self._apis)
        }

class APIValidator:
    """Validator for API requests and responses."""

    def __init__(self):
        self._validators: Dict[str, List[Callable]] = {}

    def add_validator(self, api_name: str, validator: Callable):
        """Add a validator for an API."""
        if api_name not in self._validators:
            self._validators[api_name] = []
        self._validators[api_name].append(validator)

    def validate_request(self, api_name: str, request: Any) -> bool:
        """Validate an API request."""
        if api_name not in self._validators:
            return True  # No validators means request is valid

        for validator in self._validators[api_name]:
            try:
                if not validator(request):
                    return False
            except Exception as e:
                logger.warning(f"Validator failed for {api_name}: {e}")
                return False

        return True

    def validate_response(self, response: APIResponse) -> bool:
        """Validate an API response."""
        if not isinstance(response, APIResponse):
            return False

        if response.status not in [status for status in APIResponseStatus]:
            return False

        # Additional validation logic can be added here
        return True

class InterComponentCommunicator:
    """Handles communication between components using standardized APIs."""

    def __init__(self):
        self._api_registry = APIRegistry()
        self._validator = APIValidator()
        self._communication_log: List[Dict[str, Any]] = []

    def register_component(self, name: str, api: ComponentAPI):
        """Register a component with its API."""
        self._api_registry.register_api(name, api)

        # Add default validators
        self._add_default_validators(name, api)

    def _add_default_validators(self, name: str, api: ComponentAPI):
        """Add default validators for an API."""
        if isinstance(api, PredictionAPI):
            self._validator.add_validator(name, self._validate_prediction_request)
        elif isinstance(api, DatabaseAPI):
            self._validator.add_validator(name, self._validate_database_request)

    def _validate_prediction_request(self, request: Dict[str, Any]) -> bool:
        """Validate prediction API request."""
        required_fields = ["home_team", "away_team"]
        for field in required_fields:
            if field not in request:
                return False
            if not isinstance(request[field], str) or not request[field].strip():
                return False
        return True

    def _validate_database_request(self, request: Dict[str, Any]) -> bool:
        """Validate database API request."""
        if "query" not in request:
            return False
        if not isinstance(request["query"], str) or not request["query"].strip():
            return False
        return True

    def call_api(self, component_name: str, method: str, **kwargs) -> APIResponse:
        """
        Call an API method on a component.

        Args:
            component_name: Name of the component
            method: Method to call
            **kwargs: Method arguments

        Returns:
            APIResponse from the API call
        """
        start_time = __import__('time').time()

        try:
            # Get API
            api = self._api_registry.get_api(component_name)
            if not api:
                return APIResponse(
                    status=APIResponseStatus.ERROR,
                    message=f"Component '{component_name}' not found",
                    errors=[f"No API registered for component: {component_name}"]
                )

            # Validate request
            if not self._validator.validate_request(component_name, kwargs):
                return APIResponse(
                    status=APIResponseStatus.ERROR,
                    message="Invalid request",
                    errors=["Request validation failed"]
                )

            # Call method
            if not hasattr(api, method):
                return APIResponse(
                    status=APIResponseStatus.ERROR,
                    message=f"Method '{method}' not found on component '{component_name}'",
                    errors=[f"Component does not have method: {method}"]
                )

            method_func = getattr(api, method)
            response = method_func(**kwargs)

            # Validate response
            if not self._validator.validate_response(response):
                return APIResponse(
                    status=APIResponseStatus.ERROR,
                    message="Invalid response",
                    errors=["Response validation failed"]
                )

            # Log communication
            self._log_communication(component_name, method, kwargs, response, start_time)

            return response

        except Exception as e:
            logger.error(f"API call failed: {component_name}.{method}: {e}")
            return APIResponse(
                status=APIResponseStatus.ERROR,
                message="API call failed",
                errors=[str(e)]
            )

    def _log_communication(self, component: str, method: str, request: Dict[str, Any],
                          response: APIResponse, start_time: float):
        """Log inter-component communication."""
        end_time = __import__('time').time()
        duration = end_time - start_time

        from datetime import timezone
        log_entry = {
            "timestamp": __import__('datetime').datetime.now(timezone.utc).isoformat(),
            "component": component,
            "method": method,
            "request": str(request)[:500],  # Truncate for logging
            "response_status": response.status.value,
            "duration": duration,
            "success": response.status == APIResponseStatus.SUCCESS
        }

        self._communication_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self._communication_log) > 1000:
            self._communication_log = self._communication_log[-1000:]

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        if not self._communication_log:
            return {"total_calls": 0, "success_rate": 0.0}

        total_calls = len(self._communication_log)
        successful_calls = sum(1 for log in self._communication_log if log["success"])
        avg_duration = sum(log["duration"] for log in self._communication_log) / total_calls

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "average_duration": avg_duration,
            "recent_calls": self._communication_log[-10:]  # Last 10 calls
        }

# Global instances
_api_registry = APIRegistry()
_api_validator = APIValidator()
_communicator = InterComponentCommunicator()

def get_api_registry() -> APIRegistry:
    """Get the global API registry."""
    return _api_registry

def get_api_validator() -> APIValidator:
    """Get the global API validator."""
    return _api_validator

def get_communicator() -> InterComponentCommunicator:
    """Get the global inter-component communicator."""
    return _communicator

# Convenience functions
def register_component_api(name: str, api: ComponentAPI):
    """Register a component API."""
    _api_registry.register_api(name, api)
    _communicator.register_component(name, api)

def call_component_api(component_name: str, method: str, **kwargs) -> APIResponse:
    """Call a component API method."""
    return _communicator.call_api(component_name, method, **kwargs)

def get_api_info() -> Dict[str, Any]:
    """Get information about all registered APIs."""
    return _api_registry.get_api_info()

def get_communication_stats() -> Dict[str, Any]:
    """Get inter-component communication statistics."""
    return _communicator.get_communication_stats()