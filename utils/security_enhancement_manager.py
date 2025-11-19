#!/usr/bin/env python3
"""
Security Enhancement Manager for GoalDiggers Platform
Implements comprehensive security measures including input validation, API security, and compliance.
"""

import hashlib
import hmac
import html
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Union

# Configure logging
if not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        from utils.logging_config import configure_logging  # type: ignore
        configure_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityEnhancementManager:
    """
    Comprehensive security enhancement manager.
    
    Features:
    - Input validation and sanitization
    - API rate limiting
    - Request authentication
    - XSS protection
    - SQL injection prevention
    - Security monitoring
    """
    
    def __init__(self):
        """Initialize security enhancement manager."""
        self.rate_limits = {}
        self.blocked_ips = set()
        self.security_events = []
        
        # Security configuration
        self.config = {
            'rate_limit_requests': 100,  # requests per minute
            'rate_limit_window': 60,     # seconds
            'max_input_length': 10000,   # characters
            'allowed_file_types': ['.json', '.csv', '.txt'],
            'blocked_patterns': [
                r'<script[^>]*>.*?</script>',  # XSS
                r'javascript:',                # XSS
                r'on\w+\s*=',                 # Event handlers
                r'union\s+select',            # SQL injection
                r'drop\s+table',              # SQL injection
                r'delete\s+from',             # SQL injection
            ]
        }
        
        # Initialize validation patterns
        self._initialize_validation_patterns()
        
        logger.info("🔒 Security Enhancement Manager initialized")
    
    def _initialize_validation_patterns(self):
        """Initialize validation patterns for different input types."""
        self.validation_patterns = {
            'team_name': re.compile(r'^[a-zA-Z0-9\s\-\.\']{1,50}$'),
            'league_name': re.compile(r'^[a-zA-Z0-9\s\-\.]{1,30}$'),
            'match_id': re.compile(r'^[a-zA-Z0-9\-_]{1,20}$'),
            'user_input': re.compile(r'^[a-zA-Z0-9\s\-\.\,\!\?]{1,500}$'),
            'api_key': re.compile(r'^[a-zA-Z0-9]{20,100}$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'numeric_id': re.compile(r'^\d{1,10}$')
        }
    
    def validate_input(self, input_value: Any, input_type: str = 'user_input') -> Dict[str, Any]:
        """Validate and sanitize input."""
        try:
            # Convert to string for validation
            input_str = str(input_value) if input_value is not None else ''
            
            # Check input length
            if len(input_str) > self.config['max_input_length']:
                return {
                    'valid': False,
                    'error': 'Input too long',
                    'sanitized_value': None
                }
            
            # Check for blocked patterns
            for pattern in self.config['blocked_patterns']:
                if re.search(pattern, input_str, re.IGNORECASE):
                    self._log_security_event('blocked_pattern', {
                        'pattern': pattern,
                        'input': input_str[:100]  # Log first 100 chars
                    })
                    return {
                        'valid': False,
                        'error': 'Input contains prohibited content',
                        'sanitized_value': None
                    }
            
            # Validate against specific pattern if provided
            if input_type in self.validation_patterns:
                pattern = self.validation_patterns[input_type]
                if not pattern.match(input_str):
                    return {
                        'valid': False,
                        'error': f'Invalid {input_type} format',
                        'sanitized_value': None
                    }
            
            # Sanitize the input
            sanitized_value = self._sanitize_input(input_str)
            
            return {
                'valid': True,
                'error': None,
                'sanitized_value': sanitized_value
            }
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return {
                'valid': False,
                'error': 'Validation error',
                'sanitized_value': None
            }
    
    def _sanitize_input(self, input_str: str) -> str:
        """Sanitize input to prevent XSS and other attacks."""
        # HTML escape
        sanitized = html.escape(input_str)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def check_rate_limit(self, client_id: str, endpoint: str = 'default') -> Dict[str, Any]:
        """Check if client has exceeded rate limit."""
        try:
            current_time = time.time()
            rate_key = f"{client_id}:{endpoint}"
            
            # Initialize rate limit tracking for new clients
            if rate_key not in self.rate_limits:
                self.rate_limits[rate_key] = {
                    'requests': [],
                    'blocked_until': 0
                }
            
            rate_data = self.rate_limits[rate_key]
            
            # Check if client is currently blocked
            if current_time < rate_data['blocked_until']:
                return {
                    'allowed': False,
                    'reason': 'Rate limit exceeded',
                    'retry_after': int(rate_data['blocked_until'] - current_time)
                }
            
            # Clean old requests outside the window
            window_start = current_time - self.config['rate_limit_window']
            rate_data['requests'] = [req_time for req_time in rate_data['requests'] if req_time > window_start]
            
            # Check if limit is exceeded
            if len(rate_data['requests']) >= self.config['rate_limit_requests']:
                # Block client for the rate limit window
                rate_data['blocked_until'] = current_time + self.config['rate_limit_window']
                
                self._log_security_event('rate_limit_exceeded', {
                    'client_id': client_id,
                    'endpoint': endpoint,
                    'requests_count': len(rate_data['requests'])
                })
                
                return {
                    'allowed': False,
                    'reason': 'Rate limit exceeded',
                    'retry_after': self.config['rate_limit_window']
                }
            
            # Add current request
            rate_data['requests'].append(current_time)
            
            return {
                'allowed': True,
                'reason': None,
                'requests_remaining': self.config['rate_limit_requests'] - len(rate_data['requests'])
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Allow request on error to avoid blocking legitimate users
            return {'allowed': True, 'reason': None}
    
    def validate_api_key(self, api_key: str, expected_key: Optional[str] = None) -> Dict[str, Any]:
        """Validate API key."""
        try:
            if not api_key:
                return {
                    'valid': False,
                    'error': 'API key required'
                }
            
            # Validate format
            validation_result = self.validate_input(api_key, 'api_key')
            if not validation_result['valid']:
                return {
                    'valid': False,
                    'error': 'Invalid API key format'
                }
            
            # If expected key is provided, compare
            if expected_key:
                if not hmac.compare_digest(api_key, expected_key):
                    self._log_security_event('invalid_api_key', {
                        'provided_key': api_key[:10] + '...'  # Log only first 10 chars
                    })
                    return {
                        'valid': False,
                        'error': 'Invalid API key'
                    }
            
            return {
                'valid': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return {
                'valid': False,
                'error': 'API key validation error'
            }
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash sensitive data with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password-like data
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        
        return {
            'hash': hashed.hex(),
            'salt': salt
        }
    
    def verify_hashed_data(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify hashed data."""
        try:
            expected_hash = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
            return hmac.compare_digest(expected_hash.hex(), hash_value)
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event for monitoring."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log based on severity
        if event_type in ['blocked_pattern', 'rate_limit_exceeded', 'invalid_api_key']:
            logger.warning(f"Security event: {event_type}", extra=details)
        else:
            logger.info(f"Security event: {event_type}", extra=details)
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics for monitoring."""
        if not self.security_events:
            return {'total_events': 0}
        
        # Count by event type
        event_counts = {}
        recent_events = []
        
        # Get events from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for event in self.security_events:
            event_type = event['event_type']
            event_time = datetime.fromisoformat(event['timestamp'])
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event_time > cutoff_time:
                recent_events.append(event)
        
        return {
            'total_events': len(self.security_events),
            'event_type_breakdown': event_counts,
            'recent_events_24h': len(recent_events),
            'active_rate_limits': len(self.rate_limits),
            'blocked_ips': len(self.blocked_ips)
        }
    
    def security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }


# Global singleton instance
_security_manager_instance = None

def get_security_manager() -> SecurityEnhancementManager:
    """Get global security manager instance."""
    global _security_manager_instance
    if _security_manager_instance is None:
        _security_manager_instance = SecurityEnhancementManager()
    return _security_manager_instance


def validate_input(input_value: Any, input_type: str = 'user_input') -> Dict[str, Any]:
    """Quick function to validate input."""
    manager = get_security_manager()
    return manager.validate_input(input_value, input_type)


def check_rate_limit(client_id: str, endpoint: str = 'default') -> Dict[str, Any]:
    """Quick function to check rate limit."""
    manager = get_security_manager()
    return manager.check_rate_limit(client_id, endpoint)


def secure_input_decorator(input_type: str = 'user_input'):
    """Decorator for automatic input validation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_security_manager()
            
            # Validate all string arguments
            validated_args = []
            for arg in args:
                if isinstance(arg, str):
                    validation_result = manager.validate_input(arg, input_type)
                    if not validation_result['valid']:
                        raise ValueError(f"Invalid input: {validation_result['error']}")
                    validated_args.append(validation_result['sanitized_value'])
                else:
                    validated_args.append(arg)
            
            # Validate string values in kwargs
            validated_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    validation_result = manager.validate_input(value, input_type)
                    if not validation_result['valid']:
                        raise ValueError(f"Invalid input for {key}: {validation_result['error']}")
                    validated_kwargs[key] = validation_result['sanitized_value']
                else:
                    validated_kwargs[key] = value
            
            return func(*validated_args, **validated_kwargs)
        
        return wrapper
    return decorator


def rate_limited(endpoint: str = 'default'):
    """Decorator for rate limiting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In a real implementation, you'd get the client ID from the request
            client_id = 'default_client'  # Placeholder
            
            manager = get_security_manager()
            rate_check = manager.check_rate_limit(client_id, endpoint)
            
            if not rate_check['allowed']:
                raise Exception(f"Rate limit exceeded: {rate_check['reason']}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test security manager
    manager = get_security_manager()
    
    print("🔒 Security Enhancement Manager Test")
    
    # Test input validation
    test_inputs = [
        ("Arsenal", "team_name"),
        ("<script>alert('xss')</script>", "user_input"),
        ("user@example.com", "email"),
        ("12345", "numeric_id")
    ]
    
    for input_value, input_type in test_inputs:
        result = manager.validate_input(input_value, input_type)
        print(f"\nInput: {input_value}")
        print(f"Type: {input_type}")
        print(f"Valid: {result['valid']}")
        if result['valid']:
            print(f"Sanitized: {result['sanitized_value']}")
        else:
            print(f"Error: {result['error']}")
    
    # Test rate limiting
    for i in range(5):
        rate_check = manager.check_rate_limit('test_client', 'test_endpoint')
        print(f"\nRate limit check {i+1}: {rate_check}")
    
    # Show statistics
    stats = manager.get_security_statistics()
    print(f"\nSecurity Statistics: {stats}")
