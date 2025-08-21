# -*- coding: utf-8 -*-
"""
Optimized Flask Application for Geospatial Data Processing and Visualization

This module provides a Flask web application that handles geospatial data queries,
file uploads, and real-time visualization through WebSocket connections.

Key Features:
- Real-time geospatial data processing
- WebSocket-based communication
- File upload and processing
- Session management
- Caching for improved performance
"""

import ast
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Any, Union, Tuple
from functools import lru_cache
from contextlib import contextmanager

# Third-party imports
import pyproj
import requests
from dotenv import load_dotenv
from flask import Flask, Response, stream_with_context, request, \
    render_template, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
from openai import OpenAI
from pyproj import Transformer
from shapely.geometry import Polygon, mapping, shape
from user_agents import parse
from werkzeug.utils import secure_filename

# Local imports
from core.ask_functions_agent import *
from core.agent_search_fast import id_list_of_entity_fast
from core import geo_functions


# Configuration constants
class Config:
    """Application configuration constants"""
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx',
                          'xlsx', 'csv', 'ttl'}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    GEOJSON_CACHE_SIZE = 128
    RESPONSE_CACHE_SIZE = 256
    SESSION_TIMEOUT = 3600  # 1 hour

    # File paths
    ENV_PATH = 'config/.env'
    GEOJSON_DIR = 'static/geojson'
    RESPONSES_FILE = 'responses.jsonl'
    DATA_LOG_FILE = 'static/data3.txt'
    TEST_DATA_FILE = 'evaluation_files/evaluation_fourth_dataset.jsonl'


# Initialize Flask application with optimized configuration
def create_app() -> Flask:
    """Create and configure Flask application instance"""
    app = Flask(__name__)

    # Load environment configuration
    load_dotenv(dotenv_path=Config.ENV_PATH)

    # Application configuration
    app.secret_key = os.getenv('SECRET_KEY',
                               'dev-secret-key-change-in-production')
    app.config.update({
        'UPLOAD_FOLDER': Config.UPLOAD_FOLDER,
        'MAX_CONTENT_LENGTH': Config.MAX_FILE_SIZE,
        'SESSION_COOKIE_SECURE': os.getenv('FLASK_ENV') == 'production',
        'SESSION_COOKIE_HTTPONLY': True,
        'PERMANENT_SESSION_LIFETIME': Config.SESSION_TIMEOUT,
    })

    # Create upload directory if it doesn't exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    return app


app = create_app()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize SocketIO with optimized settings
socketio = SocketIO(
    app,
    manage_session=True,  # SocketIO needs to manage session for exec() to work properly
    async_mode='threading',
    cors_allowed_origins="*"
)


# Global variables for stdout redirection (optimized)
class OutputCapture:
    """Thread-safe output capture utility"""

    def __init__(self):
        self.output = None
        self.original_stdout = sys.stdout

    @contextmanager
    def capture(self):
        """Context manager for capturing stdout"""
        # Create a new StringIO for each capture session
        self.output = StringIO()
        sys.stdout = self.output
        try:
            yield self.output
        finally:
            # Restore stdout but don't clear the output yet
            # We need to read it first with getvalue()
            sys.stdout = self.original_stdout


output_capture = OutputCapture()


# Initialize geo_functions globals (moved to initialization function)
def initialize_geo_globals():
    """Initialize global geo_functions variables"""
    geo_functions.global_id_attribute = {}
    geo_functions.global_id_geo = {}


initialize_geo_globals()


# Enhanced caching system
class GeoJSONCache:
    """Optimized GeoJSON data cache with lazy loading"""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._file_mapping = {
            '1': 'buildings_geojson.geojson',
            '2': 'land_geojson.geojson',
            '3': 'soil_maxvorstadt_geojson.geojson',
            '4': 'points_geojson.geojson',
            '5': 'lines_geojson.geojson',
        }
        self._last_modified: Dict[str, float] = {}

    @lru_cache(maxsize=Config.GEOJSON_CACHE_SIZE)
    def _load_geojson_file(self, filepath: str) -> Dict:
        """Load a single GeoJSON file with caching"""
        full_path = os.path.join(Config.GEOJSON_DIR, filepath)
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Warning: GeoJSON file {filepath} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing GeoJSON file {filepath}: {e}")
            return {}

    def get(self, key: str) -> Dict:
        """Get GeoJSON data with cache validation"""
        if key not in self._file_mapping:
            return {}

        filepath = self._file_mapping[key]
        full_path = os.path.join(Config.GEOJSON_DIR, filepath)

        try:
            current_mtime = os.path.getmtime(full_path)
            if key not in self._last_modified or current_mtime > \
                    self._last_modified[key]:
                # Clear cache for this file and reload
                self._load_geojson_file.cache_clear()
                self._cache[key] = self._load_geojson_file(filepath)
                self._last_modified[key] = current_mtime
        except OSError:
            pass  # File doesn't exist, return empty dict

        return self._cache.get(key, self._load_geojson_file(filepath))

    def preload_all(self):
        """Preload all GeoJSON files into cache"""
        for key in self._file_mapping:
            self.get(key)


# Initialize GeoJSON cache
geojson_cache = GeoJSONCache()


# Session management utilities
class SessionManager:
    """Enhanced session management with validation and cleanup"""

    @staticmethod
    def initialize_session() -> None:
        """Initialize session variables with proper defaults

        Only initializes globals_dict if it doesn't exist or is None.
        Preserves user-set bounding box across all requests.
        """
        default_globals = {
            'bounding_box_region_name': 'Munich',
            'bounding_coordinates': [48.061625, 48.248098, 11.360777,
                                     11.72291],
            'bounding_wkb': '01030000000100000005000000494C50C3B7B82640D9CEF753E3074840494C50C3B7B82640FC19DEACC11F484019E76F4221722740FC19DEACC11F484019E76F4221722740D9CEF753E3074840494C50C3B7B82640D9CEF753E3074840'
        }

        # Debug logging
        current_region = session.get('globals_dict', {}).get(
            'bounding_box_region_name', 'None') if session.get(
            'globals_dict') else 'None'
        user_set_flag = session.get('user_set_bounding_box', False)

        # Only initialize if:
        # 1. Session is completely new (globals_dict not in session)
        # 2. globals_dict is None (was explicitly cleared)
        # 3. BUT NOT if user has explicitly set a bounding box
        condition1 = 'globals_dict' not in session
        condition2 = session.get('globals_dict') is None and not user_set_flag

        print(f"\n  [DECISION LOGIC]")
        print(f"    'globals_dict' not in session: {condition1}")
        print(
            f"    globals_dict is None: {session.get('globals_dict') is None}")
        print(f"    user_set_bounding_box: {user_set_flag}")
        print(f"    Will initialize: {condition1 or condition2}")

        if condition1 or condition2:
            print(f"\n  >>> ACTION: INITIALIZING with default Munich <<<")
            print(f"      Previous region: {current_region}")
            print(f"      User set flag: {user_set_flag}")
            session['globals_dict'] = default_globals
            session['session_initialized'] = True
            # Don't mark as user-set when using defaults
            session['user_set_bounding_box'] = False
            print(f"      Set globals_dict to Munich")
            print(f"      Set user_set_bounding_box to False")
        else:
            print(f"\n  >>> ACTION: KEEPING existing session <<<")
            print(f"      Current region: {current_region}")
            print(f"      User set flag: {user_set_flag}")
            # If globals_dict is None but user had set it, don't reinitialize
            if session.get('globals_dict') is None and user_set_flag:
                print(
                    f"      WARNING: globals_dict is None but user_set_bounding_box is True")

        # Ensure session is marked as modified so it gets saved
        session.modified = True

        print(f"\n  [AFTER INIT]")
        final_region = session.get('globals_dict', {}).get(
            'bounding_box_region_name') if session.get(
            'globals_dict') else None
        print(f"    Final bounding box: {final_region}")
        print(
            f"    Final user_set_bounding_box: {session.get('user_set_bounding_box', False)}")
        print(f"    Session modified flag: {session.modified}")
        print("=" * 70)

        if 'col_name_mapping_dict' not in session:
            session['col_name_mapping_dict'] = {}

    @staticmethod
    def cleanup_uploaded_table_references() -> None:
        """Clean up uploaded table references with improved error handling"""
        try:
            global col_name_mapping_dict, fclass_dict_4_similarity, name_dict_4_similarity, all_fclass_set, all_name_set

            # Find uploaded tables
            uploaded_tables = [
                table for table in col_name_mapping_dict.keys()
                if table.startswith('uploaded_') or
                   (isinstance(col_name_mapping_dict.get(table, {}), dict) and
                    col_name_mapping_dict[table].get('graph_name',
                                                     '').startswith(
                        'uploaded_'))
            ]

            for table_name in uploaded_tables:
                # Clean up various dictionaries
                SessionManager._cleanup_table_from_dict(col_name_mapping_dict,
                                                        table_name,
                                                        "col_name_mapping_dict")
                SessionManager._cleanup_table_from_dict(
                    session.get('col_name_mapping_dict', {}), table_name,
                    "session col_name_mapping_dict")
                SessionManager._cleanup_similarity_dict(
                    fclass_dict_4_similarity, all_fclass_set, table_name,
                    "fclass_dict_4_similarity")
                SessionManager._cleanup_similarity_dict(name_dict_4_similarity,
                                                        all_name_set,
                                                        table_name,
                                                        "name_dict_4_similarity")

        except Exception as e:
            print(f"Error during session cleanup: {e}")

    @staticmethod
    def _cleanup_table_from_dict(target_dict: Dict, table_name: str,
                                 dict_name: str) -> None:
        """Helper method to clean up table from a dictionary"""
        if table_name in target_dict:
            del target_dict[table_name]
            print(f"Cleaned up {table_name} from {dict_name}")

    @staticmethod
    def _cleanup_similarity_dict(target_dict: Dict, target_set: set,
                                 table_name: str, dict_name: str) -> None:
        """Helper method to clean up similarity dictionaries"""
        if table_name in target_dict:
            target_set -= set(target_dict[table_name])
            del target_dict[table_name]
            print(f"Cleaned up {table_name} from {dict_name}")


# Input validation utilities
class InputValidator:
    """Input validation utilities for enhanced security"""

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate uploaded filename"""
        if not filename or '.' not in filename:
            return False

        # Check file extension
        extension = filename.rsplit('.', 1)[1].lower()
        if extension not in Config.ALLOWED_EXTENSIONS:
            return False

        # Additional security checks
        secure_name = secure_filename(filename)
        return secure_name == filename and len(filename) < 255

    @staticmethod
    def validate_json_input(data: Any) -> bool:
        """Validate JSON input data"""
        try:
            if isinstance(data, str):
                json.loads(data)
            elif isinstance(data, dict):
                json.dumps(data)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def sanitize_code_input(code: str) -> str:
        """Sanitize code input to prevent injection attacks"""
        # Remove potentially dangerous imports and functions
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'from\s+os\s+import',
            r'from\s+subprocess\s+import',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError(
                    f"Potentially dangerous code pattern detected: {pattern}")

        return code


# Response formatting utilities
class ResponseFormatter:
    """Enhanced response formatting with type safety"""

    @staticmethod
    def format_execution_result(result: Any, run_time: float) -> Dict[
        str, str]:
        """Format execution result for display with improved error handling"""
        if result is None:
            result = "None"

        result_str = str(result)
        error_pattern = r"An error occurred:.*\)"

        if re.search(error_pattern, result_str):
            return {'error': result_str}

        # Process result for better display
        lines = result_str.split('\n')
        processed_lines = []
        max_line_length = 150  # Maximum characters per line before truncation
        max_lines = 100  # Maximum number of lines to display

        # Process each line individually
        for i, line in enumerate(lines[:max_lines]):
            if len(line) > max_line_length:
                # Truncate long lines but keep the beginning
                processed_lines.append(f"{line[:max_line_length]}...")
            else:
                # Keep short lines intact
                processed_lines.append(line)

        # Add truncation notice if needed
        if len(lines) > max_lines:
            processed_lines.append(
                f"\n... ({len(lines) - max_lines} more lines)")

        processed_result = '\n'.join(processed_lines)

        # Calculate metadata
        total_lines = len(lines)
        total_chars = len(result_str)

        # Create a custom formatted output for frontend display
        formatted_result = f"""<code-result-block data-lines="{total_lines}" data-chars="{total_chars}" data-time="{round(run_time, 3)}">
<code-result-header>
<span class="code-result-title">📊 Code Output</span>
<span class="code-result-meta">{total_lines} lines | {total_chars} chars | {round(run_time, 3)}s</span>
</code-result-header>
<code-result-content>
{processed_result}
</code-result-content>
</code-result-block>"""
        return {'normal': formatted_result}

    @staticmethod
    def _get_result_length(result_str: str) -> Union[int, str]:
        """Get length of result with proper error handling"""
        try:
            parsed_result = ast.literal_eval(result_str)
            if parsed_result is not None:
                return len(parsed_result) if not isinstance(parsed_result,
                                                            int) else parsed_result
            return ''
        except (ValueError, SyntaxError):
            try:
                dict_result = json.loads(result_str)
                return len(dict_result)
            except (json.JSONDecodeError, TypeError):
                return f"{len(result_str)}(String)"

    @staticmethod
    def _get_attention_message(length: Union[int, str]) -> str:
        """Generate attention message based on result length"""
        if isinstance(length, str) or 'String' in str(length):
            return ''

        if int(length) > 10000:
            if int(length) == 20000:
                return 'Due to the large volume of data in your current search area, only 20,000 entries are displayed.'
            return 'Due to the large volume of data, visualization may take longer.'

        return ''

    @staticmethod
    def truncate_response(text_list: str, max_length: int = 600) -> str:
        """Truncate long responses with intelligent handling"""
        if len(str(text_list)) <= 1000:
            return text_list[:max_length]

        try:
            if ResponseFormatter._is_evaluable_list(text_list):
                length = ResponseFormatter._get_result_length(text_list)
                if isinstance(length, int) and length < 40:
                    result = ast.literal_eval(text_list)
                    return str([str(t)[:35] for t in result])
        except (ValueError, SyntaxError):
            pass

        return text_list[:max_length]

    @staticmethod
    def _is_evaluable_list(text: str) -> bool:
        """Check if text can be evaluated as a list"""
        try:
            ast.literal_eval(text)
            return True
        except (ValueError, SyntaxError):
            return False


# Code processing utilities
class CodeProcessor:
    """Enhanced code processing with better structure and error handling"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def extract_code_blocks(self, code_str: str) -> List[str]:
        """Extract Python code blocks from markdown text"""
        code_blocks = []
        parts = code_str.split("```python")

        for part in parts[1:]:
            if "```" in part:
                code_block = part.split("```")[0]
                code_blocks.append(code_block)

        return code_blocks

    def process_code_for_execution(self, code_lines: str, session: Dict,
                                   sid: str) -> str:
        """Process code string and inject necessary modifications"""
        # Split lines but keep empty lines for structure
        lines = code_lines.split('\n')
        # Filter out completely empty lines but keep indentation
        filtered_lst = [item for item in lines if item.strip()]
        lines = filtered_lst

        new_lines = []
        variable_dict = {}
        i = 0

        function_patterns = [
            'geo_filter(', 'id_list_of_entity(', 'id_list_of_entity_fast(',
            'add_or_subtract_entities(', 'area_filter(', 'set_bounding_box(',
            'traffic_navigation('
        ]

        while i < len(lines):
            line = lines[i].strip()  # Strip only when processing
            i = self._process_single_line(line, lines, i, new_lines,
                                          variable_dict,
                                          function_patterns, session, sid)

        # Handle the last line
        self._handle_final_line(new_lines)

        return '\n'.join(new_lines)

    def _process_single_line(self, line: str, all_lines: List[str],
                             current_index: int,
                             new_lines: List[str], variable_dict: Dict,
                             function_patterns: List[str], session: Dict,
                             sid: str) -> int:
        """Process a single line of code"""
        has_assignment = '=' in line
        has_function_call = any(func in line for func in function_patterns)

        if has_assignment and has_function_call:
            return self._handle_assignment_with_function(
                line, all_lines, current_index, new_lines, variable_dict,
                function_patterns, session, sid
            )
        elif has_function_call and not has_assignment:
            return self._handle_function_without_assignment(
                line, all_lines, current_index, new_lines,
                function_patterns, session, sid
            )
        else:
            new_lines.append(line)
            return current_index + 1

    def _handle_assignment_with_function(self, line: str, all_lines: List[str],
                                         current_index: int,
                                         new_lines: List[str],
                                         variable_dict: Dict,
                                         function_patterns: List[str],
                                         session: Dict, sid: str) -> int:
        """Handle line with variable assignment and function call"""
        variable_name = line.split('=')[0].strip()
        # Pass the original line from all_lines to preserve formatting
        original_line = all_lines[current_index].strip()
        full_function, next_index = self._collect_multiline_function(
            original_line,
            all_lines,
            current_index)

        # Inject bounding box if needed
        full_function = self._inject_bounding_box_if_needed(full_function,
                                                            session)

        variable_dict[variable_name] = full_function
        comment_index = self._find_comment_for_line(all_lines,
                                                    all_lines[current_index],
                                                    session)

        new_lines.append(full_function)
        new_lines.append(
            f"send_data({variable_name}['geo_map'], 'map', '{comment_index}', sid='{sid}')")

        return next_index

    def _handle_function_without_assignment(self, line: str,
                                            all_lines: List[str],
                                            current_index: int,
                                            new_lines: List[str],
                                            function_patterns: List[str],
                                            session: Dict, sid: str) -> int:
        """Handle function call without variable assignment"""
        # Pass the original line from all_lines to preserve formatting
        original_line = all_lines[current_index].strip()
        full_function, next_index = self._collect_multiline_function(
            original_line,
            all_lines,
            current_index)

        # Inject bounding box if needed
        full_function = self._inject_bounding_box_if_needed(full_function,
                                                            session)

        comment_index = self._find_comment_for_line(all_lines,
                                                    all_lines[current_index],
                                                    session)

        new_lines.append(f"temp_result = {full_function}")
        new_lines.append(
            f"send_data(temp_result['geo_map'], 'map', '{comment_index}', sid='{sid}')")

        return next_index

    def _collect_multiline_function(self, initial_line: str,
                                    all_lines: List[str],
                                    start_index: int) -> Tuple[str, int]:
        """Collect a complete multiline function call"""
        full_function = initial_line
        open_parens = initial_line.count('(')
        close_parens = initial_line.count(')')
        current_index = start_index + 1

        while current_index < len(all_lines) and open_parens > close_parens:
            line = all_lines[current_index].strip() if current_index < len(
                all_lines) else ""
            full_function += '\n' + line
            open_parens += line.count('(')
            close_parens += line.count(')')
            current_index += 1

        return full_function, current_index

    def _inject_bounding_box_if_needed(self, function_call: str,
                                       session: Dict) -> str:
        """Inject bounding box parameter if needed"""
        if 'id_list_of_entity_fast(' not in function_call or 'bounding_box=' in function_call:
            return function_call

        last_paren_index = function_call.rfind(')')
        if last_paren_index == -1:
            return function_call

        open_paren_index = function_call.find('(')
        content_between_parens = function_call[
                                 open_paren_index + 1:last_paren_index].strip()
        separator = ", " if content_between_parens else ""

        return (
                function_call[:last_paren_index] +
                f"{separator}bounding_box=session['globals_dict']" +
                function_call[last_paren_index:]
        )

    def _find_comment_for_line(self, all_lines: List[str], target_line: str,
                               session: Dict) -> str:
        """Find appropriate comment for a line of code"""
        special_char = '#><;' if session.get('template') else '#'

        comment_positions = [
            (i, line.strip()) for i, line in enumerate(all_lines)
            if line.strip().startswith("#")
        ]

        try:
            target_index = all_lines.index(target_line)
        except ValueError:
            return ""

        for idx, (pos, comment) in enumerate(comment_positions):
            if idx + 1 < len(comment_positions):
                next_pos = comment_positions[idx + 1][0]
                if pos < target_index < next_pos:
                    return comment.replace(special_char, '').replace("'",
                                                                     '').strip()
            else:
                if pos < target_index:
                    return comment.replace(special_char, '').replace("'",
                                                                     '').strip()

        return ""

    def _handle_final_line(self, lines: List[str]) -> None:
        """Handle processing of the final line"""
        if not lines:
            return

        last_line = lines[-1]
        excluded_patterns = ['=', 'send_data', 'id_list_explain(', '#']

        if not any(pattern in last_line for pattern in excluded_patterns):
            lines[-1] = f"print_function({last_line.strip()})"


# Initialize code processor
code_processor = CodeProcessor(SessionManager())


# Global functions for use in exec() context
def send_data(data, mode="data", index="", sid=''):
    """Global wrapper for WebSocketManager.send_data to be accessible in exec() context"""
    return WebSocketManager.send_data(data, mode, index, sid)


def print_function(var_name):
    """Custom print function with enhanced formatting for better display."""
    try:
        var_str = str(var_name)

        # Handle different data types for better formatting
        if isinstance(var_name, (list, tuple)):
            if len(var_name) > 100:
                # For large lists/tuples, show summary
                print(f"[{type(var_name).__name__}] Length: {len(var_name)}")
                print(f"First 10 items: {var_name[:10]}")
                if len(var_name) > 10:
                    print(f"... and {len(var_name) - 10} more items")
            else:
                # For smaller lists, pretty print them
                import pprint
                pprint.pprint(var_name, width=80, compact=False)
        elif isinstance(var_name, dict):
            if len(var_name) > 50:
                # For large dicts, show summary
                print(f"[Dictionary] {len(var_name)} keys")
                print(f"First 10 keys: {list(var_name.keys())[:10]}")
            else:
                # For smaller dicts, pretty print them
                import pprint
                pprint.pprint(var_name, width=80, compact=False)
        elif len(var_str) > 4000:
            # For very long strings or other types
            print(
                f"[{type(var_name).__name__}] Output too long (length: {len(var_str)})")
            print(f"First 500 characters:")
            print(var_str[:500])
            print("...")
        else:
            # For normal output, just print it
            print(var_name)
    except Exception as e:
        print(f"Error in print_function: {e}")
        print(f"Type: {type(var_name).__name__}")


# Alias for backward compatibility
print_process = print_function


# WebSocket utilities
class WebSocketManager:
    """Enhanced WebSocket communication management"""

    @staticmethod
    def send_data(data: Any, mode: str = "data", index: str = "",
                  sid: str = '') -> None:
        """Send data via WebSocket with improved error handling"""
        try:
            target_labels = []

            if mode == "map" and not isinstance(data, str) and data:
                if isinstance(data, dict) and 'target_label' in data:
                    target_labels = data.pop('target_label')
                data = WebSocketManager._convert_polygons_to_geojson(data)

            if sid:
                socketio.emit('text', {
                    mode: data,
                    'index': index,
                    'target_label': target_labels
                }, room=sid)
            else:
                print('No session ID provided for WebSocket communication')

        except Exception as e:
            print(f"Error sending WebSocket data: {e}")

    @staticmethod
    def _convert_polygons_to_geojson(polygons_dict: Dict) -> Dict:
        """Convert polygon objects to GeoJSON format"""
        if not isinstance(polygons_dict, dict):
            return polygons_dict

        try:
            return {key: mapping(polygon) for key, polygon in
                    polygons_dict.items()}
        except Exception as e:
            print(f"Error converting polygons to GeoJSON: {e}")
            return polygons_dict


# File processing utilities
class FileProcessor:
    """Enhanced file processing with better error handling"""

    @staticmethod
    def read_column_names(text: str) -> Dict:
        """Extract column names from GeoJSON text using GPT"""
        ask_prompt = """
I will give you a piece of geojson text, and you need to return a json, tell me the key names that can represent the elements category, name, geom:
Do not use the 'id' key as any of the key values
Do not give key like 'key.label', do not give higher level key, just give the direct label

If you are not sure, give the most likely answer

give me a short logic reasoning before giving the result
please give the json format like below:
```json
{
"category": "Example Category",
"name": "name",
"geom": "geometry",
}
```
        """
        try:
            result_json = general_gpt_without_memory(
                query=text,
                ask_prompt=ask_prompt,
                json_mode='json_few_shot',
                verbose=True
            )
            return result_json
        except Exception as e:
            print(f"Error reading column names: {e}")
            return {}

    @staticmethod
    def find_key_values(json_obj: Union[Dict, List], target_key: str) -> List:
        """Find all values for a specific key in nested JSON structure"""
        result = []

        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if key == target_key:
                    if isinstance(value, str):
                        result.append(value.replace("_", " "))
                    else:
                        result.append(value)
                elif isinstance(value, (dict, list)):
                    result.extend(
                        FileProcessor.find_key_values(value, target_key))

        elif isinstance(json_obj, list):
            for item in json_obj:
                result.extend(FileProcessor.find_key_values(item, target_key))

        return result

    @staticmethod
    def get_geojson_epsg(file_path: str) -> Union[int, str]:
        """Extract EPSG code from GeoJSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            crs = data.get('crs', {}).get('properties', {}).get('name')

            if crs:
                try:
                    epsg = pyproj.CRS(crs).to_epsg()
                    return epsg if epsg else f"Cannot parse EPSG code: {crs}"
                except Exception as e:
                    return f"CRS parsing failed: {e}"
            else:
                return "GeoJSON does not provide CRS information"

        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def convert_to_wkt_4326(geojson_list: List[Dict], epsg: str = '3857') -> \
            List[str]:
        """Convert GeoJSON coordinates to WKT format in EPSG:4326"""
        try:
            transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326",
                                               always_xy=True)
            wkt_list = []

            for geojson in geojson_list:
                geom_type = geojson.get("type")
                coordinates = geojson.get("coordinates")

                if not coordinates:
                    raise ValueError("Invalid GeoJSON: Missing coordinates.")

                transformed_coords = FileProcessor._transform_coordinates(
                    coordinates, transformer)
                shapely_geom = shape(
                    {"type": geom_type, "coordinates": transformed_coords})
                wkt_list.append(shapely_geom.wkt)

            return wkt_list

        except Exception as e:
            print(f"Error converting to WKT: {e}")
            return []

    @staticmethod
    def _transform_coordinates(coords: Union[List, Tuple],
                               transformer: Transformer) -> Union[List, Tuple]:
        """Recursively transform coordinates"""
        if isinstance(coords[0], (int, float)):
            x, y = coords[:2]
            return transformer.transform(x, y)

        return [FileProcessor._transform_coordinates(c, transformer) for c in
                coords]


# IP location utilities
class IPLocationService:
    """IP location service with caching and error handling"""

    @staticmethod
    @lru_cache(maxsize=1000)
    def query_ip_location(ip: str) -> str:
        """Query IP location with caching"""
        try:
            ip = ip.strip()
            url = f"http://ip-api.com/json/{ip}"

            response = requests.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "success":
                region = data.get("regionName", "")
                city = data.get("city", "")
                isp = data.get("isp", "")
                return f"{region}  {city}  {isp}".strip()
            else:
                return ip

        except Exception as e:
            print(f"Error querying IP location: {e}")
            return ip


# Custom print function with optimization (duplicate definition removed - see line 682)


# Flask route handlers with enhanced error handling and validation
# Commented out to fix session persistence issue
# The initialization is now done only in home() route
# @app.before_request
# def before_request():
#     """Initialize session and perform pre-request setup"""
#     SessionManager.initialize_session()
#     SessionManager.cleanup_uploaded_table_references()
#     print('Session initialized and cleaned up')


@socketio.on('join')
def on_join(data: Dict) -> None:
    """Handle WebSocket join event with validation"""
    try:
        session['sid'] = request.sid
        WebSocketManager.send_data(session['sid'], 'sid', sid=session['sid'])
    except Exception as e:
        print(f"Error in WebSocket join: {e}")


@app.route('/')
def home():
    """Main homepage route with enhanced session management"""
    try:
        print("\n" + "#" * 70)
        print("[HOME_ROUTE] STARTING")
        print(f"  Session ID: {session.get('sid', 'Unknown')}")
        bbox_before = session.get('globals_dict', {}).get(
            'bounding_box_region_name') if session.get(
            'globals_dict') else None
        print(f"  Bounding box BEFORE init: {bbox_before}")
        print(
            f"  User set flag BEFORE: {session.get('user_set_bounding_box', False)}")
        print("#" * 70)

        print("Initializing application")
        del_uploaded_sql()  # Assuming this function exists

        # Initialize session only on home page load
        print("\n[HOME_ROUTE] Calling SessionManager.initialize_session()...")
        SessionManager.initialize_session()
        SessionManager.cleanup_uploaded_table_references()
        print('[HOME_ROUTE] Session initialized and cleaned up')

        bbox_after = session.get('globals_dict', {}).get(
            'bounding_box_region_name') if session.get(
            'globals_dict') else None
        print(f"\n[HOME_ROUTE] AFTER INIT:")
        print(f"  Bounding box: {bbox_after}")
        print(
            f"  User set flag: {session.get('user_set_bounding_box', False)}")

        # Initialize other session variables with validation
        session.setdefault('file_path', '')
        session.setdefault('ip_', request.remote_addr or 'unknown')
        session.setdefault('uploaded_indication', None)
        session.setdefault('sid', '')
        session.setdefault('template', False)
        session.setdefault('history', [])

        # Parse user agent with error handling
        user_agent_string = request.headers.get('User-Agent')
        if user_agent_string:
            try:
                user_agent = parse(user_agent_string)
                session.update({
                    'os': user_agent.os.family,
                    'browser': user_agent.browser.family,
                    'device_type': (
                        'Mobile' if user_agent.is_mobile else
                        'Tablet' if user_agent.is_tablet else
                        'Desktop' if user_agent.is_pc else 'Unknown'
                    )
                })
            except Exception as e:
                print(f"Error parsing user agent: {e}")
                session.update({'os': 'Unknown', 'browser': 'Unknown',
                                'device_type': 'Unknown'})
        else:
            session.update({'os': None, 'browser': None, 'device_type': None})

        return render_template('index.html')

    except Exception as e:
        print(f"Error in home route: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/introduction')
def introduction():
    """Introduction page route"""
    return render_template('introduction.html')


@app.route('/geojson/<key>')
def send_geojson(key: str):
    """Send cached GeoJSON data with validation"""
    try:
        # Validate key parameter
        if not key or not re.match(r'^[1-5]$', key):
            return jsonify({'error': 'Invalid key parameter'}), 400

        geojson_data = geojson_cache.get(key)
        return jsonify(geojson_data)

    except Exception as e:
        print(f"Error serving GeoJSON for key {key}: {e}")
        return jsonify({'error': 'Failed to load GeoJSON data'}), 500


@app.route('/question')
def question():
    """Question page route"""
    return render_template('question.html')


@app.route('/thank_you')
def thank_you():
    """Thank you page route"""
    return render_template('thank_you.html')


@app.route('/submit-qu', methods=['POST'])
def submit_questionnaire():
    """Handle questionnaire submission with enhanced validation and logging"""
    try:
        start_time = request.form.get('start_time')
        end_time = time.time()
        ip_address = request.remote_addr or 'unknown'

        # Validate start_time
        if start_time:
            try:
                float(start_time)
            except ValueError:
                return jsonify({'error': 'Invalid start_time format'}), 400

        # Filter and validate form data
        answers = {
            key: value for key, value in request.form.items()
            if key != 'start_time' and len(str(value)) < 10000
            # Prevent overly long inputs
        }

        response_data = {
            'ip_address': ip_address,
            'start_time': start_time,
            'end_time': end_time,
            'timestamp': datetime.now().isoformat(),
            'answers': answers
        }

        # Atomic file writing
        temp_file = Config.RESPONSES_FILE + '.tmp'
        with open(temp_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(response_data, ensure_ascii=False) + '\n')

        # Rename to final file (atomic operation)
        os.rename(temp_file, Config.RESPONSES_FILE)

        return redirect(url_for('thank_you'))

    except Exception as e:
        print(f"Error in questionnaire submission: {e}")
        return jsonify({'error': 'Failed to submit questionnaire'}), 500


@app.route('/debug_mode', methods=['POST'])
def debug_mode():
    """Toggle debug mode with validation"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid request data'}), 400

        message = str(data.get('message', '')).lower()
        session['template'] = (message == 'debug')

        return jsonify({"text": True, "debug_enabled": session['template']})

    except Exception as e:
        print(f"Error in debug mode toggle: {e}")
        return jsonify({'error': 'Failed to toggle debug mode'}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with enhanced validation and security"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate filename
        if not InputValidator.validate_filename(file.filename):
            return jsonify({'error': 'Invalid filename or file type'}), 400

        # Secure filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)

        # Check if file already exists
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1

        file.save(file_path)

        return jsonify({
            'file_path': file_path,
            'original_filename': file.filename,
            'saved_filename': os.path.basename(file_path)
        })

    except Exception as e:
        print(f"Error in file upload: {e}")
        return jsonify({'error': 'Failed to upload file'}), 500


@app.route('/read_file', methods=['POST'])
def read_file():
    """Read and analyze uploaded file with enhanced error handling"""
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': 'Missing file_path parameter'}), 400

        file_path = data.get('file_path')
        session['file_path'] = file_path

        # Validate file path
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return jsonify({'error': 'File not found'}), 404

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > Config.MAX_FILE_SIZE:
            return jsonify({'error': 'File too large'}), 413

        # Read file content with encoding detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1200)  # Limit initial read
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read(1200)
            except Exception as e:
                return jsonify({'error': f'Cannot decode file: {e}'}), 400

        # Analyze content
        result_json = FileProcessor.read_column_names(content)

        if result_json:
            result_json['epsg'] = FileProcessor.get_geojson_epsg(file_path)
            return jsonify(result_json)
        else:
            return jsonify({'error': 'Failed to analyze file content'}), 500

    except Exception as e:
        print(f"Error reading file: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/read_result', methods=['POST'])
def read_result():
    """Process file analysis results with enhanced validation"""
    try:
        data = request.get_json()
        if not data or 'file_path' not in data or 'table' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        file_path = data['file_path']
        table_name = str(data['table']).lower().strip()

        # Validate table name
        if not table_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$',
                                          table_name):
            return jsonify({'error': 'Invalid table name'}), 400

        # Check for existing table names
        if (table_name in session.get('col_name_mapping_dict', {}) or
                table_name in globals().get('all_fclass_set', set()) or
                table_name in globals().get('all_name_set', set())):
            return jsonify({'error': 'Table name already exists'}), 400

        # Read and validate file content
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Validate JSON content
        if not InputValidator.validate_json_input(content):
            return jsonify({'error': 'Invalid JSON content'}), 400

        content_json = json.loads(content)

        # Process form data
        store_json = {}
        required_fields = ['category', 'name', 'geom', 'epsg']

        for field in required_fields:
            if field in data and field != 'table' and field != 'file_path':
                if field == 'epsg':
                    continue

                values = FileProcessor.find_key_values(content_json,
                                                       data[field])
                if not values:
                    return jsonify(
                        {'error': f'No data found for field: {field}'}), 400

                store_json[field] = values
                print(f"Found {len(values)} items for {field}")

        # Convert coordinates
        if 'geom' in store_json and 'epsg' in data:
            try:
                store_json['geom'] = FileProcessor.convert_to_wkt_4326(
                    store_json['geom'], data['epsg']
                )
            except Exception as e:
                return jsonify(
                    {'error': f'Coordinate conversion failed: {e}'}), 400

        # Rename category to fclass
        if 'category' in store_json:
            store_json['fclass'] = store_json.pop('category')

        # Add OSM IDs
        store_json['osm_id'] = list(range(len(store_json.get('fclass', []))))

        # Create table and update global dictionaries
        try:
            create_table_from_json(store_json,
                                   table_name)  # Assuming this function exists

            # Update session and global mappings
            session['col_name_mapping_dict'][table_name] = {}

            if 'col_name_mapping_dict' in globals():
                globals()['col_name_mapping_dict'][table_name] = {
                    "osm_id": "osm_id",
                    "fclass": "fclass",
                    "name": "name",
                    "select_query": f"SELECT uploaded_{table_name} AS source_table, fclass,name,osm_id,geom",
                    "graph_name": f"uploaded_{table_name}"
                }

            # Update similarity dictionaries
            if 'fclass_dict_4_similarity' in globals() and 'all_fclass_set' in globals():
                fclass_set = ids_of_attribute(
                    table_name)  # Assuming this function exists
                globals()['fclass_dict_4_similarity'][table_name] = fclass_set
                globals()['all_fclass_set'].update(fclass_set)

            if 'name_dict_4_similarity' in globals() and 'all_name_set' in globals():
                name_set = ids_of_attribute(table_name,
                                            'name')  # Assuming this function exists
                globals()['name_dict_4_similarity'][table_name] = name_set
                globals()['all_name_set'].update(name_set)

            return jsonify({'success': True, 'table_name': table_name})

        except Exception as e:
            print(f"Error creating table: {e}")
            return jsonify({'error': 'Failed to create table from data'}), 500

    except Exception as e:
        print(f"Error processing results: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/submit', methods=['POST'])
def submit():
    """Main submission handler with enhanced code processing and error handling"""
    print("\n" + "*" * 70)
    print("[SUBMIT_ROUTE] STARTING")
    print(f"  Session ID: {session.get('sid', 'Unknown')}")
    bbox = session.get('globals_dict', {}).get(
        'bounding_box_region_name') if session.get('globals_dict') else None
    print(f"  Current bounding box: {bbox}")
    print(f"  User set flag: {session.get('user_set_bounding_box', False)}")
    print("*" * 70)

    try:
        # Validate request data
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Extract and validate parameters
        user_input = json_data.get('text', '').strip()
        messages = json_data.get('messages', [])
        sid = json_data.get('sid', '').strip()
        current_mode = json_data.get('currentMode', '').strip()
        agent_mode = json_data.get('agentMode',
                                   'analyzer').strip()  # New parameter

        if not user_input:
            return jsonify({'error': 'No input text provided'}), 400

        # Validate and sanitize input
        try:
            user_input = InputValidator.sanitize_code_input(user_input)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        # Process the submission
        def process_submission():
            """Generator function for processing submission with streaming response"""
            processed_response = []

            try:
                # Check if explainer mode is selected
                if agent_mode == 'explainer':
                    # Use explainer agent for Cypher-based processing
                    yield from _process_explainer_submission(
                        user_input, messages, sid, processed_response
                    )
                else:
                    # Use traditional analyzer mode
                    yield from _process_code_submission(
                        user_input, messages, sid, current_mode,
                        processed_response
                    )
            except Exception as e:
                print(f"Error in code processing: {e}")
                yield f"Error: {str(e)}\n"

        return Response(
            stream_with_context(process_submission()),
            mimetype='text/plain'
        )

    except Exception as e:
        print(f"Error in submit route: {e}")
        return jsonify({'error': 'Internal server error'}), 500


def _process_explainer_submission(user_input: str, messages: List[Dict],
                                  sid: str,
                                  processed_response: List[Dict]):
    """Process submission using explainer agent for Cypher-based queries"""
    try:
        from core.ask_functions_agent import explainer_agent

        # Show processing status
        yield "Processing your question with Explainer Agent...\n\n"

        # Call explainer agent
        result = explainer_agent(user_input, messages)

        # Yield the result
        yield result

        # Add to processed response
        processed_response.append({
            'role': 'assistant',
            'content': result
        })

        # Send final response via WebSocket
        WebSocketManager.send_data(processed_response, sid=sid)

        # Log interaction
        _log_interaction(user_input, sid, [result], processed_response)

    except Exception as e:
        error_msg = f"Error in explainer submission: {str(e)}"
        print(error_msg)
        yield f"\n\nError: {error_msg}\n"


def _process_code_submission(user_input: str, messages: List[Dict], sid: str,
                             current_mode: str,
                             processed_response: List[Dict]):
    """Process code submission with improved structure"""
    yield_list = []

    try:
        if session.get('template'):
            # Template mode - direct code execution
            code_list = [user_input]
            yield_list.append(user_input)

            # Print user's direct code input
            print("\n" + "=" * 80)
            print("USER DIRECT CODE INPUT (Template Mode):")
            print("=" * 80)
            print(user_input)
            print("=" * 80 + "\n")
        else:
            # AI processing mode
            messages.append(message_template('user',
                                             user_input))  # Assuming this function exists

            # Get AI response
            try:
                chat_response = chat_single(messages,
                                            "stream")  # Assuming this function exists

                # Process streaming response and yield in real-time
                total_buffer = ""
                for char_output in _process_streaming_response_generator(
                        chat_response, current_mode
                ):
                    if isinstance(char_output, tuple):
                        # This is the final result
                        total_buffer = char_output[0]
                        yield_list.extend(char_output[1])
                    else:
                        # This is a character to yield
                        yield char_output
                        yield_list.append(char_output)

                # Replace function calls if not in reasoning mode
                if current_mode != 'reasoning':
                    total_buffer = total_buffer.replace("id_list_of_entity(",
                                                        'id_list_of_entity_fast(')

                total_buffer = total_buffer.replace('print(',
                                                    'print_function(')

                processed_response.append(
                    {'role': 'assistant', 'content': total_buffer})
                messages.append({'role': 'assistant', 'content': total_buffer})

                # Extract code blocks
                code_list = []
                if ("```python" in total_buffer and
                        ".env" not in total_buffer and
                        "pip install" not in total_buffer):
                    code_list.extend(
                        code_processor.extract_code_blocks(total_buffer))

                    # Print LLM returned code for debugging
                    if code_list:
                        print("\n" + "=" * 80)
                        print("LLM RETURNED CODE:")
                        print("=" * 80)
                        for i, code in enumerate(code_list):
                            print(f"\n[Code Block {i + 1}]:")
                            print("-" * 40)
                            print(code)
                            print("-" * 40)
                        print("=" * 80 + "\n")

            except Exception as e:
                yield f"Error in AI processing: {str(e)}\n"
                return

        # Execute code blocks
        for code_block in code_list:
            yield from _execute_code_block(code_block, sid, messages,
                                           processed_response)

        # Send final response
        WebSocketManager.send_data(processed_response, sid=sid)

        # Log interaction
        _log_interaction(user_input, sid, yield_list, processed_response)

        # Yield all collected output
        for item in yield_list:
            yield item

    except Exception as e:
        print(f"Error in code submission processing: {e}")
        yield f"Error: {str(e)}\n"


def _process_streaming_response_generator(chat_response, current_mode: str):
    """Process streaming response from AI with improved handling - generator version"""
    total_buffer = ""
    yield_list = []
    line_buffer = ""
    chunk_num = 0
    in_code_block = False

    try:
        for chunk in chat_response:
            if chunk and chunk.choices[0].delta.content:
                char = ("\n" + chunk.choices[
                    0].delta.content) if chunk_num == 0 else chunk.choices[
                    0].delta.content
                chunk_num += 1
                total_buffer += char
                line_buffer += char

                # Handle code block detection
                if "```python".startswith(line_buffer) and not in_code_block:
                    in_code_block = True
                    line_buffer = ""
                    continue
                elif "```".startswith(line_buffer) and in_code_block:
                    in_code_block = False
                    line_buffer = ""
                    continue

                # Yield non-code content character by character
                if (
                        not in_code_block and line_buffer) or line_buffer.startswith(
                        '#'):
                    output_char = char.replace('#', '#><;').replace("'", '')
                    yield output_char  # Yield character immediately

                # Reset line buffer on newline
                if '\n' in char:
                    line_buffer = ""

    except Exception as e:
        print(f"Error processing streaming response: {e}")

    # Return final result as tuple
    yield (total_buffer, yield_list)


def _process_streaming_response(chat_response, current_mode: str) -> Tuple[
    str, List[str]]:
    """Process streaming response from AI with improved handling - non-generator version for backward compatibility"""
    total_buffer = ""
    yield_list = []
    line_buffer = ""
    chunk_num = 0
    in_code_block = False

    try:
        for chunk in chat_response:
            if chunk and chunk.choices[0].delta.content:
                char = ("\n" + chunk.choices[
                    0].delta.content) if chunk_num == 0 else chunk.choices[
                    0].delta.content
                chunk_num += 1
                total_buffer += char
                line_buffer += char

                # Handle code block detection
                if "```python".startswith(line_buffer) and not in_code_block:
                    in_code_block = True
                    line_buffer = ""
                    continue
                elif "```".startswith(line_buffer) and in_code_block:
                    in_code_block = False
                    line_buffer = ""
                    continue

                # Collect non-code content
                if (
                        not in_code_block and line_buffer) or line_buffer.startswith(
                    '#'):
                    yield_list.append(
                        char.replace('#', '#><;').replace("'", ''))

                # Reset line buffer on newline
                if '\n' in char:
                    line_buffer = ""

    except Exception as e:
        print(f"Error processing streaming response: {e}")

    return total_buffer, yield_list


def _execute_code_block(code: str, sid: str, messages: List[Dict],
                        processed_response: List[Dict]):
    """Execute a single code block with enhanced error handling"""
    try:
        yield "\n\n`Code running...`\n"

        # Handle matplotlib plots
        plt_show = "plt.show()" in code
        if plt_show:
            filename = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            code = _modify_matplotlib_code(code, filename)

        # Process code for execution
        processed_code = code_processor.process_code_for_execution(code,
                                                                   session,
                                                                   sid)

        # Print processed code (not captured in execution result since it's before output capture)
        print(f"\n{'=' * 60}")
        print("EXECUTING PROCESSED CODE:")
        print('-' * 60)
        print(processed_code)
        print('=' * 60)

        # Execute code with output capture
        start_time = time.time()

        print(f"\n[EXECUTE_CODE_BLOCK] About to exec() processed code")
        print(
            f"  Code snippet: {processed_code[:150] if len(processed_code) > 150 else processed_code}")
        bbox_before_exec = session.get('globals_dict', {}).get(
            'bounding_box_region_name') if session.get(
            'globals_dict') else None
        print(f"  Bounding box BEFORE exec: {bbox_before_exec}")

        with output_capture.capture() as output:
            try:
                exec(processed_code, globals())

                bbox_after_exec = session.get('globals_dict', {}).get(
                    'bounding_box_region_name') if session.get(
                    'globals_dict') else None
                # print(f"\n[EXECUTE_CODE_BLOCK] AFTER exec()")
                # print(f"  Bounding box: {bbox_after_exec}")


            except Exception as e:
                exc_info = traceback.format_exc()
                if session.get('template'):
                    print(str(e))
                print(f"An error occurred: \n{exc_info}")

        session.modified = True
        end_time = time.time()
        run_time = end_time - start_time

        # Get execution result
        code_result = str(output.getvalue().replace('\00', ''))

        # Debug: Print what we're about to send
        print(f"\n[CODE_RESULT] Length: {len(code_result)}")
        if code_result:
            print(f"[CODE_RESULT] First 200 chars: {code_result[:200]}")
        else:
            print("[CODE_RESULT] Empty - no output from code execution")

        # Handle matplotlib output
        if plt_show and "An error occurred: " not in code_result:
            if not os.path.exists(f"static/{filename}"):
                filename = 'plot_20240711162140.png'  # Fallback

            code_result = f'![matplotlib_diagram](/static/{filename} "matplotlib_diagram")'
            yield code_result

        # Format and yield result
        formatted_result = ResponseFormatter.format_execution_result(
            code_result, run_time)
        result_to_send = list(formatted_result.values())[0]


        yield result_to_send

        # Handle errors
        if 'error' in formatted_result:
            return

        # Add to message history
        short_result = ResponseFormatter.truncate_response(code_result)
        result_message = {"role": "user",
                          "content": f"code_result:{short_result}"}
        messages.append(result_message)
        processed_response.append(
            {'role': 'user', 'content': f"code_result:{short_result}"})

    except Exception as e:
        print(f"Error executing code block: {e}")
        yield f"Error executing code: {str(e)}\n"


def _modify_matplotlib_code(code: str, filename: str) -> str:
    """Modify matplotlib code for web display"""
    code = code.replace(
        "import matplotlib.pyplot as plt",
        "import matplotlib.pyplot as plt\nfrom matplotlib.font_manager import FontProperties\nfont = FontProperties(fname=r'static\\msyh.ttc')\n"
    )
    code = code.replace("plt.show()",
                        f"plt.tight_layout()\nplt.savefig('static/{filename}')")
    code = code.replace("plt.figure(figsize=(10, 6))",
                        "plt.figure(figsize=(5, 5))")
    return code


def _log_interaction(user_input: str, sid: str, yield_list: List[str],
                     processed_response: List[Dict]):
    """Log user interaction with enhanced data structure"""
    try:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': sid,
            'ip_address': session.get('ip_', 'unknown'),
            'user_input': user_input,
            'os': session.get('os'),
            'browser': session.get('browser'),
            'device': session.get('device_type'),
            'yield_list': yield_list,
            'response': processed_response
        }

        formatted_data = json.dumps(log_data, indent=2, ensure_ascii=False)

        with open(Config.DATA_LOG_FILE, 'a', encoding='utf-8') as file:
            file.write(formatted_data + '\n')

    except Exception as e:
        print(f"Error logging interaction: {e}")


@app.route('/get_test_data', methods=['GET'])
def get_test_data():
    """Read test data from JSONL file with enhanced error handling"""
    try:
        test_data = []

        if not os.path.exists(Config.TEST_DATA_FILE):
            return jsonify({
                'success': False,
                'error': 'Test file not found'
            }), 404

        with open(Config.TEST_DATA_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    if 'query_result' in data:
                        test_data.append({
                            'id': line_num,
                            'query': data['query_result']
                        })

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue

        return jsonify({
            'success': True,
            'total': len(test_data),
            'data': test_data
        })

    except Exception as e:
        print(f"Error reading test data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "error": "The file is too large. Maximum file size is 500MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


# Application initialization
def initialize_application():
    """Initialize application with preloading and setup"""
    print("Initializing application...")

    # Preload GeoJSON data for better performance
    geojson_cache.preload_all()
    print("GeoJSON data preloaded")

    # Initialize global variables
    initialize_geo_globals()
    print("Global variables initialized")

    print("Application initialization complete")


if __name__ == '__main__':
    initialize_application()

    # Run with optimized settings
    socketio.run(
        app,
        allow_unsafe_werkzeug=True,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 9090)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )