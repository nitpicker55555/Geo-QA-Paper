"""
Security utilities for the Flask application.

This module provides security-related functions including input validation,
sanitization, rate limiting, and security headers.
"""

import hashlib
import hmac
import re
import time
import secrets
from typing import Dict, List, Optional, Pattern, Set, Any
from collections import defaultdict, deque
from functools import wraps
import threading


class InputValidator:
    """Enhanced input validation with security focus."""
    
    # Compiled regex patterns for performance
    FILENAME_PATTERN: Pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
    SQL_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)', re.IGNORECASE),
        re.compile(r'(\'|\"|;|--|\*|\/\*|\*\/)', re.IGNORECASE),
        re.compile(r'(\bOR\b.*=.*\b|\bAND\b.*=.*\b)', re.IGNORECASE)
    ]
    
    XSS_PATTERNS: List[Pattern] = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL)
    ]
    
    CODE_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r'import\s+(os|subprocess|sys)', re.IGNORECASE),
        re.compile(r'from\s+(os|subprocess|sys)\s+import', re.IGNORECASE),
        re.compile(r'__import__\s*\(', re.IGNORECASE),
        re.compile(r'eval\s*\(', re.IGNORECASE),
        re.compile(r'exec\s*\(', re.IGNORECASE),
        re.compile(r'open\s*\(', re.IGNORECASE),
        re.compile(r'file\s*\(', re.IGNORECASE)
    ]
    
    @classmethod
    def validate_filename(cls, filename: str, max_length: int = 255) -> bool:
        """
        Validate uploaded filename for security.
        
        Args:
            filename: Filename to validate
            max_length: Maximum allowed length
            
        Returns:
            True if filename is safe, False otherwise
        """
        if not filename or len(filename) > max_length:
            return False
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check filename pattern
        if not cls.FILENAME_PATTERN.match(filename):
            return False
        
        return True
    
    @classmethod
    def validate_extension(cls, filename: str, allowed_extensions: Set[str]) -> bool:
        """
        Validate file extension.
        
        Args:
            filename: Filename to check
            allowed_extensions: Set of allowed extensions
            
        Returns:
            True if extension is allowed, False otherwise
        """
        if not filename or '.' not in filename:
            return False
        
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in allowed_extensions
    
    @classmethod
    def detect_sql_injection(cls, input_string: str) -> bool:
        """
        Detect potential SQL injection attempts.
        
        Args:
            input_string: String to check
            
        Returns:
            True if potential SQL injection detected, False otherwise
        """
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if pattern.search(input_string):
                return True
        return False
    
    @classmethod
    def detect_xss(cls, input_string: str) -> bool:
        """
        Detect potential XSS attempts.
        
        Args:
            input_string: String to check
            
        Returns:
            True if potential XSS detected, False otherwise
        """
        for pattern in cls.XSS_PATTERNS:
            if pattern.search(input_string):
                return True
        return False
    
    @classmethod
    def detect_code_injection(cls, code_string: str) -> bool:
        """
        Detect potential code injection attempts.
        
        Args:
            code_string: Code string to check
            
        Returns:
            True if potential code injection detected, False otherwise
        """
        for pattern in cls.CODE_INJECTION_PATTERNS:
            if pattern.search(code_string):
                return True
        return False
    
    @classmethod
    def sanitize_html(cls, input_string: str) -> str:
        """
        Basic HTML sanitization.
        
        Args:
            input_string: String to sanitize
            
        Returns:
            Sanitized string
        """
        # Replace HTML entities
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }
        
        result = input_string
        for char, entity in replacements.items():
            result = result.replace(char, entity)
        
        return result
    
    @classmethod
    def validate_json_structure(cls, data: Any, required_fields: List[str] = None) -> bool:
        """
        Validate JSON structure and required fields.
        
        Args:
            data: Data to validate
            required_fields: List of required field names
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            return False
        
        if required_fields:
            for field in required_fields:
                if field not in data:
                    return False
        
        return True


class RateLimiter:
    """Rate limiting implementation with sliding window."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_minutes: Window size in minutes
        """
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for identifier.
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            
        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()
        
        with self.lock:
            # Clean old requests outside the window
            while (self.requests[identifier] and 
                   current_time - self.requests[identifier][0] > self.window_seconds):
                self.requests[identifier].popleft()
            
            # Check if under limit
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(current_time)
                return True
            
            return False
    
    def get_remaining_requests(self, identifier: str) -> int:
        """
        Get remaining requests for identifier.
        
        Args:
            identifier: Client identifier
            
        Returns:
            Number of remaining requests
        """
        current_time = time.time()
        
        with self.lock:
            # Clean old requests
            while (self.requests[identifier] and 
                   current_time - self.requests[identifier][0] > self.window_seconds):
                self.requests[identifier].popleft()
            
            return max(0, self.max_requests - len(self.requests[identifier]))
    
    def reset_limit(self, identifier: str) -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Client identifier
        """
        with self.lock:
            self.requests[identifier].clear()


class SessionSecurityManager:
    """Enhanced session security management."""
    
    def __init__(self, secret_key: str):
        """
        Initialize session security manager.
        
        Args:
            secret_key: Secret key for signing
        """
        self.secret_key = secret_key.encode('utf-8')
    
    def generate_csrf_token(self) -> str:
        """
        Generate CSRF token.
        
        Returns:
            CSRF token string
        """
        return secrets.token_urlsafe(32)
    
    def validate_csrf_token(self, token: str, session_token: str) -> bool:
        """
        Validate CSRF token.
        
        Args:
            token: Token to validate
            session_token: Session's CSRF token
            
        Returns:
            True if valid, False otherwise
        """
        return hmac.compare_digest(token, session_token)
    
    def sign_data(self, data: str) -> str:
        """
        Sign data with HMAC.
        
        Args:
            data: Data to sign
            
        Returns:
            Signature string
        """
        return hmac.new(
            self.secret_key, 
            data.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """
        Verify data signature.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        expected_signature = self.sign_data(data)
        return hmac.compare_digest(signature, expected_signature)
    
    def generate_session_id(self) -> str:
        """
        Generate secure session ID.
        
        Returns:
            Session ID string
        """
        return secrets.token_urlsafe(32)


class SecurityHeaders:
    """Security headers management."""
    
    DEFAULT_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        ),
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=(), "
            "accelerometer=(), ambient-light-sensor=()"
        )
    }
    
    @classmethod
    def apply_headers(cls, response, additional_headers: Dict[str, str] = None):
        """
        Apply security headers to response.
        
        Args:
            response: Flask response object
            additional_headers: Additional headers to apply
        """
        headers = cls.DEFAULT_HEADERS.copy()
        
        if additional_headers:
            headers.update(additional_headers)
        
        for header, value in headers.items():
            response.headers[header] = value
        
        return response


def rate_limit(max_requests: int = 100, window_minutes: int = 1, 
               key_func: Optional[callable] = None):
    """
    Rate limiting decorator.
    
    Args:
        max_requests: Maximum requests per window
        window_minutes: Window size in minutes
        key_func: Function to generate rate limit key
    """
    limiter = RateLimiter(max_requests, window_minutes)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request, jsonify
            
            # Determine identifier
            if key_func:
                identifier = key_func()
            else:
                identifier = request.remote_addr or 'unknown'
            
            if not limiter.is_allowed(identifier):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': window_minutes * 60
                }), 429
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def csrf_protect(session_manager: SessionSecurityManager):
    """
    CSRF protection decorator.
    
    Args:
        session_manager: Session security manager instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request, session, jsonify
            
            if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
                # Check CSRF token
                token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
                session_token = session.get('csrf_token')
                
                if not token or not session_token:
                    return jsonify({'error': 'CSRF token missing'}), 400
                
                if not session_manager.validate_csrf_token(token, session_token):
                    return jsonify({'error': 'CSRF token invalid'}), 400
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def secure_headers(additional_headers: Dict[str, str] = None):
    """
    Security headers decorator.
    
    Args:
        additional_headers: Additional headers to apply
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            return SecurityHeaders.apply_headers(response, additional_headers)
        return wrapper
    return decorator


class FileSecurityValidator:
    """File upload security validation."""
    
    DANGEROUS_EXTENSIONS = {
        'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar',
        'sh', 'py', 'pl', 'php', 'asp', 'aspx', 'jsp'
    }
    
    MAGIC_NUMBERS = {
        b'\x89PNG\r\n\x1a\n': 'png',
        b'\xff\xd8\xff': 'jpg',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif',
        b'%PDF': 'pdf',
        b'PK\x03\x04': 'zip',  # Also covers xlsx, docx
        b'\xd0\xcf\x11\xe0': 'doc',  # MS Office
    }
    
    @classmethod
    def validate_file_content(cls, file_path: str, expected_extension: str) -> bool:
        """
        Validate file content matches extension using magic numbers.
        
        Args:
            file_path: Path to file
            expected_extension: Expected file extension
            
        Returns:
            True if content matches extension, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)  # Read first 16 bytes
            
            for magic_bytes, file_type in cls.MAGIC_NUMBERS.items():
                if header.startswith(magic_bytes):
                    return file_type == expected_extension.lower()
            
            # For text files, check if content is valid text
            if expected_extension.lower() in ['txt', 'csv', 'json', 'xml']:
                return cls._is_text_file(file_path)
            
            return True  # Allow other file types
        
        except (OSError, IOError):
            return False
    
    @classmethod
    def _is_text_file(cls, file_path: str) -> bool:
        """Check if file contains valid text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read first 1KB as text
            return True
        except UnicodeDecodeError:
            return False
        except (OSError, IOError):
            return False
    
    @classmethod
    def is_safe_extension(cls, extension: str) -> bool:
        """
        Check if file extension is safe.
        
        Args:
            extension: File extension to check
            
        Returns:
            True if extension is safe, False otherwise
        """
        return extension.lower() not in cls.DANGEROUS_EXTENSIONS
    
    @classmethod
    def scan_file_for_malicious_content(cls, file_path: str) -> List[str]:
        """
        Scan file for potentially malicious content.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List of detected threats
        """
        threats = []
        
        try:
            # For text-based files, scan content
            if cls._is_text_file(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for script tags
                if re.search(r'<script[^>]*>.*?</script>', content, re.IGNORECASE | re.DOTALL):
                    threats.append('Script tags detected')
                
                # Check for suspicious URLs
                if re.search(r'https?://[^\s<>"]+', content):
                    threats.append('External URLs detected')
                
                # Check for SQL keywords
                sql_keywords = ['DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET']
                for keyword in sql_keywords:
                    if keyword in content.upper():
                        threats.append(f'SQL keyword detected: {keyword}')
                        break
            
            # Check file size (prevent zip bombs, etc.)
            import os
            file_size = os.path.getsize(file_path)
            if file_size > 500 * 1024 * 1024:  # 500MB
                threats.append('File too large')
        
        except Exception as e:
            threats.append(f'Error scanning file: {str(e)}')
        
        return threats


# Global instances
default_rate_limiter = RateLimiter(max_requests=100, window_minutes=1)
upload_rate_limiter = RateLimiter(max_requests=10, window_minutes=1)
api_rate_limiter = RateLimiter(max_requests=1000, window_minutes=1)