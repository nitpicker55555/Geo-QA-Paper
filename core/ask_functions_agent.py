# -*- coding: utf-8 -*-
"""
Optimized Agent Functions Module

This module provides optimized functions for spatial data processing and similarity calculations
with improved performance, caching, and memory efficiency.

Key Optimizations:
- Caching system for repeated calculations
- Optimized data structures and algorithms
- Better memory management
- Improved error handling and type safety
- Modular design with clear separation of concerns
"""

import json
import os
import random
import re
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
import time

import spacy
from flask import session

# Import from unified RAG service
from services.rag_service import (
    find_word_in_sentence,
    calculate_similarity_openai,
    calculate_similarity_chroma
)
from services.chat_py import *
from utils.levenshtein import are_strings_similar
from .geo_functions import *
from .bounding_box import find_boundbox
from .neo4j_mapper import (
    get_neo4j_mapper,
    get_fclass_dict_for_similarity,
    get_name_dict_for_similarity,
    get_all_fclass_set,
    get_all_name_set
)


# ============================================================================
# Global Variables and Configuration
# ============================================================================

# Load spaCy model only once
nlp = spacy.load('en_core_web_sm')

# Global dictionaries with type hints
global_paring_dict: Dict[str, Any] = {}
fclass_dict: Dict[str, Any] = {}
name_dict: Dict[str, Any] = {}

# Global sets and dictionaries for similarity matching
# These will now be populated from Neo4j when needed
all_fclass_set: Set[str] = set()
all_name_set: Set[str] = set()
fclass_dict_4_similarity: Dict[str, Set[str]] = {}
name_dict_4_similarity: Dict[str, Set[str]] = {}

# Flag to track if Neo4j data has been loaded
_neo4j_data_loaded = False


# ============================================================================
# Caching and Performance Optimization
# ============================================================================

class PerformanceCache:
    """
    Centralized caching system for expensive operations
    """
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with LRU eviction"""
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        self._access_times.clear()


# Global cache instance
performance_cache = PerformanceCache()


def cached(cache_key_func: Optional[Callable] = None):
    """
    Decorator for caching function results
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Check cache
            cached_result = performance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            performance_cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


# ============================================================================
# Optimized Data Structure Initialization
# ============================================================================

def initialize_similarity_dictionaries(use_db: bool = True, use_neo4j: bool = True) -> None:
    """
    Initialize similarity dictionaries with optimized performance
    
    Args:
        use_db: If True, query database for actual values. If False, use empty sets.
        use_neo4j: If True, use Neo4j for data. If False, use PostgreSQL directly.
    """
    global fclass_dict_4_similarity, name_dict_4_similarity, all_fclass_set, all_name_set, _neo4j_data_loaded
    
    if not use_db:
        # Initialize with empty sets for testing without database
        for table_name in col_name_mapping_dict:
            fclass_dict_4_similarity[table_name] = set()
            if table_name != 'soil':
                name_dict_4_similarity[table_name] = set()
        return
    
    if use_neo4j:
        try:
            # Load from Neo4j
            mapper = get_neo4j_mapper()
            fclass_dict_4_similarity, name_dict_4_similarity = mapper.build_similarity_dictionaries()
            all_fclass_set = mapper.get_all_fclass_values()
            all_name_set = mapper.get_all_name_values()
            _neo4j_data_loaded = True
            print("Loaded similarity dictionaries from Neo4j")
            return
        except Exception as e:
            print(f"Failed to load from Neo4j: {e}, falling back to PostgreSQL")
    
    # Original PostgreSQL implementation
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for table_name in col_name_mapping_dict:
            # Submit fclass processing
            futures.append(
                executor.submit(_process_table_attributes, table_name, 'fclass')
            )
            
            # Submit name processing (except for soil)
            if table_name != 'soil':
                futures.append(
                    executor.submit(_process_table_attributes, table_name, 'name')
                )
        
        # Collect results
        for future in as_completed(futures):
            try:
                table_name, attribute_type, attribute_set = future.result()
                
                if attribute_type == 'fclass':
                    fclass_dict_4_similarity[table_name] = attribute_set
                    all_fclass_set.update(attribute_set)
                else:
                    name_dict_4_similarity[table_name] = attribute_set
                    all_name_set.update(attribute_set)
            except Exception as e:
                print(f"Error processing {table_name} {attribute_type}: {e}")


def _process_table_attributes(table_name: str, attribute_type: str) -> Tuple[str, str, Set[str]]:
    """
    Process table attributes in separate thread
    """
    attribute_set = ids_of_attribute(table_name, attribute_type)
    return table_name, attribute_type, attribute_set


def ensure_neo4j_data_loaded():
    """
    Ensure Neo4j data is loaded before using similarity dictionaries
    """
    global _neo4j_data_loaded
    if not _neo4j_data_loaded:
        initialize_similarity_dictionaries(use_db=True, use_neo4j=True)


# Simple initialization similar to reference file
# This populates the dictionaries without database access during import
try:
    # Initialize with empty sets first
    for table_name in col_name_mapping_dict:
        fclass_dict_4_similarity[table_name] = set()
        if table_name != 'soil':
            name_dict_4_similarity[table_name] = set()
    
    # The actual data will be loaded lazily when needed
    # or can be explicitly initialized by calling initialize_similarity_dictionaries(use_db=True)
except Exception as e:
    print(f"Error during module initialization: {e}")


# ============================================================================
# Optimized Utility Functions
# ============================================================================

@lru_cache(maxsize=1000)
def limit_total_words(text_tuple: Tuple[str, ...], max_length: int = 10000) -> List[str]:
    """
    Limit total word count with caching
    """
    total_length = 0
    result = []
    
    for item in text_tuple:
        current_length = len(item)
        if total_length + current_length > max_length:
            break
        result.append(item)
        total_length += current_length
    
    return result


def safe_print(*args) -> None:
    """
    Safe printing function with error handling
    """
    try:
        for arg in args:
            print(arg, end=' ')
        print()  # New line
    except Exception as e:
        print(f"Print error: {e}")


class QueryProcessor:
    """
    Optimized query processing class with caching and better error handling
    """
    
    def __init__(self):
        self.string_cache: Dict[str, str] = {}
    
    @cached()
    def process_string(self, s: str) -> str:
        """
        Process string by removing digits and leading colons
        """
        if not isinstance(s, str):
            return str(s)
        
        processed = re.sub(r'\d+', '', s)
        if processed.startswith(':'):
            processed = processed[1:]
        return processed.strip()
    
    @cached()
    def has_middle_space(self, s: str) -> bool:
        """
        Check if string has space in the middle (not at ends)
        """
        if not isinstance(s, str) or len(s.strip()) < 2:
            return False
        
        stripped = s.strip()
        return ' ' in stripped[1:-1]
    
    @cached()
    def extract_numbers(self, s: str) -> int:
        """
        Extract numbers from string with size modifiers
        """
        if not isinstance(s, str):
            return 1
        
        numbers = re.findall(r'\d+', s)
        modifier = -1 if 'small' in s.lower() else 1
        
        return int(numbers[0]) * modifier if numbers else modifier


# Global query processor instance
query_processor = QueryProcessor()


# ============================================================================
# Optimized Similarity and Matching Functions
# ============================================================================

class SimilarityCalculator:
    """
    Optimized similarity calculation with caching and parallel processing
    """
    
    def __init__(self):
        self.similarity_cache: Dict[str, List[str]] = {}
    
    @cached(lambda self, string, lst: f"partial_match:{string}:{hash(tuple(sorted(lst)))}")
    def find_partial_matches(self, string: str, lst: List[str]) -> Set[str]:
        """
        Find partial string matches in list with optimization
        """
        if not string or not lst:
            return set()
        
        string_lower = string.lower()
        matches = set()
        
        # Use set operations for faster lookup
        lst_set = {str(item).lower(): item for item in lst}
        
        # Exact match first
        if string_lower in lst_set:
            matches.add(lst_set[string_lower])
            return matches
        
        # Partial matches
        string_words = set(string_lower.split())
        for item_lower, item_original in lst_set.items():
            item_words = set(item_lower.split())
            if string_words.intersection(item_words):
                matches.add(item_original)
        
        return matches
    
    @cached()
    def calculate_enhanced_similarity(
        self, 
        query: str, 
        given_list: List[str], 
        table_name: str,
        similarity_threshold: float = 0.7
    ) -> List[str]:
        """
        Enhanced similarity calculation with multiple strategies
        """
        if not query or not given_list:
            return []
        
        # Strategy 1: Exact and partial matches
        partial_matches = self.find_partial_matches(query, given_list)
        if partial_matches:
            return list(partial_matches)
        
        # Strategy 2: Vector similarity with caching
        try:
            vector_matches = calculate_similarity_chroma(
                query=query, 
                give_list=given_list, 
                mode='fclass'
            )[0]
            
            if vector_matches:
                return list(vector_matches)
        
        except Exception as e:
            safe_print(f"Vector similarity error: {e}")
        
        # Strategy 3: Fallback to OpenAI similarity
        try:
            openai_matches = calculate_similarity_openai(table_name, query)
            if openai_matches:
                return openai_matches
        
        except Exception as e:
            safe_print(f"OpenAI similarity error: {e}")
        
        return []


# Global similarity calculator instance
similarity_calc = SimilarityCalculator()


# ============================================================================
# Optimized Main Processing Functions
# ============================================================================

class FeatureMatcher:
    """
    Optimized feature matching with better error handling and performance
    """
    
    def __init__(self):
        self.column_cache: Dict[str, str] = {}
    
    @cached()
    def determine_column_name(self, statement: str, table_name: str) -> str:
        """
        Determine appropriate column name with caching
        """
        if not statement:
            return 'fclass'
        
        statement_lower = statement.lower()
        
        # Quick keyword checks
        if any(word in statement_lower for word in ['name', 'call']):
            return 'name'
        
        if any(word in statement_lower for word in ['large', 'small', 'big']):
            return 'area_num#'
        
        # Check table columns
        try:
            col_names = get_column_names(table_name)
            for col_name in col_names:
                if col_name in statement.split():
                    return col_name
        except Exception as e:
            safe_print(f"Column lookup error: {e}")
        
        return 'fclass'
    
    @cached()
    def judge_numeric_comparison(self, query_type: str) -> Union[int, bool]:
        """
        Judge if query involves numeric comparison
        """
        if not isinstance(query_type, str):
            return False
        
        query_lower = query_type.lower()
        
        if any(word in query_lower for word in ['higher', 'lower', 'bigger', 'smaller']):
            numbers = query_processor.extract_numbers(query_type)
            if any(word in query_lower for word in ['higher', 'bigger']):
                return abs(numbers)
            else:
                return -abs(numbers)
        
        return False
    
    def process_feature_matching(
        self, 
        query_feature: str, 
        table_name: str, 
        verbose: bool = False,
        bounding_box: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Optimized feature matching with comprehensive error handling
        """
        try:
            query_feature = query_feature.strip() if isinstance(query_feature, str) else str(query_feature)
        except Exception as e:
            raise ValueError(f"Invalid query feature: {e}")
        
        # Handle compound features
        query_list = query_feature.split(" and ") if " and " in query_feature else [query_feature]
        
        # Initialize result structure
        match_result: Dict[str, Any] = {
            'non_area_col': {'fclass': set(), 'name': set()},
            'area_num': None
        }
        
        for query in query_list:
            if not query.strip():
                continue
                
            try:
                self._process_single_query(
                    query.strip(), 
                    table_name, 
                    match_result, 
                    verbose, 
                    bounding_box
                )
            except Exception as e:
                safe_print(f"Error processing query '{query}': {e}")
                continue
        
        if verbose:
            safe_print(f"Match result: {match_result}")
        
        return match_result
    
    def _process_single_query(
        self, 
        query: str, 
        table_name: str, 
        match_result: Dict[str, Any],
        verbose: bool,
        bounding_box: Optional[Dict]
    ) -> None:
        """
        Process single query with optimized logic
        """
        col_name = self.determine_column_name(query, table_name)
        
        if '#' not in col_name:  # Non-area column processing
            self._process_non_area_column(
                query, 
                table_name, 
                col_name, 
                match_result, 
                verbose, 
                bounding_box
            )
        else:  # Area-related query
            match_result['area_num'] = query_processor.extract_numbers(query)
    
    def _process_non_area_column(
        self, 
        query: str, 
        table_name: str, 
        col_name: str,
        match_result: Dict[str, Any],
        verbose: bool,
        bounding_box: Optional[Dict]
    ) -> None:
        """
        Process non-area column queries with optimization
        """
        if col_name not in match_result['non_area_col']:
            match_result['non_area_col'][col_name] = set()
        
        # Check for table name similarity
        if are_strings_similar(query, table_name):
            match_result['non_area_col'][col_name].add('all')
            return
        
        # Get attribute list
        try:
            given_list = ids_of_attribute(
                table_name, 
                col_name, 
                bounding_box_coordinates=bounding_box
            )
        except Exception as e:
            safe_print(f"Error getting attributes: {e}")
            return
        
        # Clean query
        cleaned_query = self._clean_query(query, table_name, col_name)
        
        # Check for numeric comparison
        num_compare = self.judge_numeric_comparison(cleaned_query)
        if num_compare:
            compared_list = self._compare_numbers(given_list, num_compare)
            match_result['non_area_col'][col_name].update(compared_list)
            if verbose:
                safe_print(f"Numeric comparison results: {compared_list}")
            return
        
        # Find matches using optimized similarity calculation
        matches = similarity_calc.calculate_enhanced_similarity(
            cleaned_query, 
            given_list, 
            table_name
        )
        
        if matches:
            match_result['non_area_col'][col_name].update(set(matches))
            if verbose:
                safe_print(f"Found matches: {matches}")
    
    @cached()
    def _clean_query(self, query: str, table_name: str, col_name: str) -> str:
        """
        Clean query by removing common stop words and table references
        """
        stop_words = ['named', 'is', 'which', 'where', 'has', 'call', 'called', table_name, col_name]
        
        for word in stop_words:
            query = query.replace(word, '')
        
        return query.strip()
    
    @cached()
    def _compare_numbers(self, lst: List[str], target_num: int) -> Set[str]:
        """
        Compare numbers in list with target number
        """
        result_set = set()
        
        for item in lst:
            if str(item).isnumeric():
                item_num = int(item)
                if (target_num > 0 and item_num > abs(target_num)) or \
                   (target_num < 0 and item_num < abs(target_num)):
                    result_set.add(item)
        
        return result_set


# Global feature matcher instance
feature_matcher = FeatureMatcher()


# ============================================================================
# Optimized Geographic Processing Functions
# ============================================================================

class GeographicProcessor:
    """
    Optimized geographic processing with better error handling
    """
    
    @cached()
    def judge_geo_relation(self, query: str) -> Optional[Dict[str, Union[str, int]]]:
        """
        Judge geographic relation with caching and optimization
        """
        if not query:
            return None
        
        query_clean = query.replace('with', '').strip().lower()
        
        # Quick pattern matching - check longer patterns first to avoid substring matches
        relation_patterns = [
            ('contains', {'type': 'contains', 'num': 0}),
            ('intersects', {'type': 'intersects', 'num': 0}),
            ('around', {'type': 'buffer', 'num': 100}),
            ('under', {'type': 'contains', 'num': 0}),
            ('in', {'type': 'in', 'num': 0}),
            ('on', {'type': 'in', 'num': 0})
        ]
        
        for pattern, relation in relation_patterns:
            if pattern in query_clean:
                return relation
        
        # Extract distance for buffer operations
        distance_match = re.search(r'(\d+)\s*m', query_clean)
        if distance_match and any(word in query_clean for word in ['around', 'near', 'close']):
            return {'type': 'buffer', 'num': int(distance_match.group(1))}
        
        # Fallback to AI processing for complex queries
        try:
            return self._process_complex_geo_relation(query)
        except Exception as e:
            safe_print(f"Error processing geo relation: {e}")
            return None
    
    def _process_complex_geo_relation(self, query: str) -> Optional[Dict[str, Union[str, int]]]:
        """
        Process complex geographic relations using AI
        """
        ask_prompt = """Analyze the geographic relationship in this query. Return JSON with:
        {"geo_calculations": {"exist": true/false, "type": "buffer/in/contains/intersects", "num": distance}}
        
        Examples:
        - "100m around" -> {"geo_calculations": {"exist": true, "type": "buffer", "num": 100}}
        - "contains" -> {"geo_calculations": {"exist": true, "type": "contains", "num": 0}}
        - "no relation" -> {"geo_calculations": {"exist": false}}
        """
        
        messages = [
            message_template('system', ask_prompt),
            message_template('user', query)
        ]
        
        try:
            result = chat_single(messages, 'json')
            json_result = json.loads(result)
            
            if 'geo_calculations' in json_result and json_result['geo_calculations']['exist']:
                geo_calc = json_result['geo_calculations']
                return {
                    'type': geo_calc.get('type', 'intersects'),
                    'num': geo_calc.get('num', 0)
                }
        
        except Exception as e:
            safe_print(f"AI geo processing error: {e}")
        
        return None
    
    def process_geographic_filter(
        self, 
        query: str, 
        id_list_subject: Union[str, Dict], 
        id_list_object: Union[str, Dict],
        bounding_box: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process geographic filtering with optimization
        """
        # Convert string inputs to ID lists if necessary
        if isinstance(id_list_subject, str):
            id_list_subject = self._get_entity_id_list(id_list_subject, bounding_box)
        
        if isinstance(id_list_object, str):
            id_list_object = self._get_entity_id_list(id_list_object, bounding_box)
        
        # Check for negation
        versa_sign = self._find_negation(query)
        if versa_sign:
            query = re.sub(r'\b(not|no|never|none)\b', '', query, flags=re.IGNORECASE).strip()
        
        # Get geographic relation
        geo_relation = self.judge_geo_relation(query)
        if not geo_relation:
            return {'error': 'No geographical relation found in query.'}
        
        # Perform geographic calculation
        try:
            geo_result = geo_calculate(
                id_list_subject, 
                id_list_object,
                geo_relation['type'], 
                geo_relation['num'],
                versa_sign=versa_sign,
                bounding_box=bounding_box
            )
            
            # Add target label information
            if 'subject' in geo_result and 'id_list' in geo_result['subject']:
                target_labels = self._get_labels_from_ids(geo_result['subject']['id_list'])
                geo_result['geo_map']['target_label'] = list(target_labels)
            
            return geo_result
        
        except Exception as e:
            safe_print(f"Geographic calculation error: {e}")
            return {'error': f'Geographic calculation failed: {e}'}
    
    @cached()
    def _find_negation(self, text: str) -> bool:
        """
        Find negation in text using spaCy with caching
        """
        if not text:
            return False
        
        try:
            doc = nlp(text)
            return any(token.dep_ == 'neg' for token in doc)
        except Exception as e:
            safe_print(f"Negation detection error: {e}")
            return False
    
    def _get_entity_id_list(self, entity: str, bounding_box: Optional[Dict] = None) -> Dict:
        """
        Get entity ID list with error handling
        """
        try:
            return id_list_of_entity(entity, bounding_box=bounding_box)
        except Exception as e:
            safe_print(f"Entity ID list error: {e}")
            return {'id_list': {}, 'table_name': ''}
    
    @cached()
    def _get_labels_from_ids(self, id_list: Dict[str, Any]) -> Set[str]:
        """
        Extract labels from ID list with caching
        """
        if not isinstance(id_list, dict):
            return set()
        
        labels = set()
        for key in id_list.keys():
            if isinstance(key, str) and key.count('_') >= 2:
                # Extract label from ID format
                parts = key.split('_')
                if len(parts) >= 3:
                    labels.add('_'.join(parts[:2]))
        
        return labels


# Global geographic processor instance
geo_processor = GeographicProcessor()


# ============================================================================
# Name Similarity Functions
# ============================================================================

def name_cosin_list(query: str, name_set: Set[str]) -> Tuple[List[str], bool]:
    """
    Find similar names using cosine similarity
    
    Args:
        query: The name to search for
        name_set: Set of names to search within
        
    Returns:
        Tuple of (list of matching names, success flag)
    """
    try:
        if not query or not name_set:
            return [], False
            
        # Convert set to list for calculate_similarity_chroma
        name_list = list(name_set)
        
        # Use the existing calculate_similarity_chroma function
        matches, success = calculate_similarity_chroma(
            query=query,
            give_list=name_list,
            mode='name',
            results_num=10  # Return top 10 matches
        )
        
        # Filter matches to only include those in the original set
        if matches:
            filtered_matches = [m for m in matches if m in name_set]
            return filtered_matches, True
        
        return [], False
        
    except Exception as e:
        safe_print(f"Error in name_cosin_list: {e}")
        return [], False


def find_keys_by_values(d: Dict, elements: Union[List, Set]) -> Dict:
    """
    Find keys in dictionary that contain matching values from elements
    
    Args:
        d: Dictionary to search in
        elements: Elements to find in dictionary values
        
    Returns:
        Dictionary with keys and their matched elements
    """
    result = {}
    for key, values in d.items():
        matched_elements = [element for element in elements if element in values]
        if matched_elements:
            result[key] = matched_elements
    return result


def merge_dicts(dict_list: Union[Dict, List[Dict]]) -> Dict:
    """
    Merge multiple dictionaries into one
    
    Args:
        dict_list: Single dictionary or list of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    if isinstance(dict_list, dict):
        dict_list = [dict_list]
    
    result = {}
    for d in dict_list:
        if not isinstance(d, dict):
            continue
        for key, subdict in d.items():
            if key not in result:
                result[key] = subdict.copy() if isinstance(subdict, dict) else subdict
            else:
                if isinstance(result[key], dict) and isinstance(subdict, dict):
                    result[key].update(subdict)
                else:
                    result[key] = subdict
    return result


def judge_table(table_name: str) -> bool:
    """
    Judge if a table name is valid
    
    Args:
        table_name: Name of the table to check
        
    Returns:
        True if table is valid, False otherwise
    """
    return table_name in col_name_mapping_dict


# ============================================================================
# Optimized Entity Processing Functions
# ============================================================================

class EntityProcessor:
    """
    Optimized entity processing with improved performance and error handling
    """
    
    def __init__(self):
        self.table_cache: Dict[str, str] = {}
    
    @cached()
    def judge_table_name(self, query: str) -> Optional[Dict[str, str]]:
        """
        Judge appropriate table name with caching and optimization
        """
        if not query or not isinstance(query, str):
            return None
        
        query_lower = query.lower()
        
        # Soil-related keywords
        soil_keywords = ['planting', 'potatoes', 'tomatoes', 'strawberr', 'agriculture', 'soil', 'farming']
        if any(keyword in query_lower for keyword in soil_keywords):
            return {'database': 'soil'}
        
        # Check exact table name matches
        query_words = set(query_lower.split())
        
        # Check similar table names first
        if hasattr(self, 'similar_table_name_dict'):
            for similar_name, actual_name in similar_table_name_dict.items():
                if similar_name in query_words:
                    return {'database': actual_name}
        
        # Check direct table names
        for table_name in col_name_mapping_dict:
            if table_name in query_words:
                return {'database': table_name}
        
        # Special case for greenery
        if 'greenery' in query_lower:
            return {'database': 'area'}
        
        return None
    
    def process_entity_list(
        self, 
        query: str, 
        verbose: bool = False, 
        bounding_box: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process entity list with optimized performance and comprehensive error handling
        """
        if not query:
            return None
        
        # Normalize query
        query = query.lower().replace("strasse", 'straße')
        
        # Get bounding box from session if not provided
        if bounding_box is None:
            bounding_box = session.get('globals_dict')
        
        # Update global dictionaries if bounding box is specified
        if bounding_box:
            self._update_global_dictionaries_for_bbox(bounding_box)
        
        # Check for specific table restrictions
        table_restriction = self.judge_table_name(query)
        if table_restriction:
            safe_print(f"Table restriction detected: {table_restriction['database']}")
            query += f" (Limited to table: {table_restriction['database']})"
        
        try:
            # Use optimized agent-based search
            return self._agent_based_search(query, verbose, bounding_box)
        
        except Exception as e:
            safe_print(f"Error in entity processing: {e}")
            return None
    
    def _update_global_dictionaries_for_bbox(self, bounding_box: Dict) -> None:
        """
        Update global dictionaries for bounding box with parallel processing
        """
        global all_fclass_set, all_name_set, fclass_dict_4_similarity, name_dict_4_similarity
        
        all_fclass_set.clear()
        all_name_set.clear()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for table_name in col_name_mapping_dict:
                # Submit fclass processing
                futures.append(
                    executor.submit(
                        ids_of_attribute, 
                        table_name, 
                        'fclass', 
                        bounding_box_coordinates=bounding_box
                    )
                )
                
                # Submit name processing (except for soil)
                if table_name != 'soil':
                    futures.append(
                        executor.submit(
                            ids_of_attribute, 
                            table_name, 
                            'name', 
                            bounding_box_coordinates=bounding_box
                        )
                    )
            
            # Collect results
            future_to_table = {}
            for i, future in enumerate(futures):
                table_idx = i // 2 if i % 2 == 0 else i // 2
                table_name = list(col_name_mapping_dict.keys())[table_idx]
                attr_type = 'fclass' if i % 2 == 0 else 'name'
                future_to_table[future] = (table_name, attr_type)
            
            for future in as_completed(futures):
                try:
                    result_set = future.result()
                    table_name, attr_type = future_to_table[future]
                    
                    if attr_type == 'fclass':
                        fclass_dict_4_similarity[table_name] = result_set
                        all_fclass_set.update(result_set)
                    else:
                        name_dict_4_similarity[table_name] = result_set
                        all_name_set.update(result_set)
                
                except Exception as e:
                    safe_print(f"Error updating dictionaries: {e}")
    
    def _agent_based_search(
        self, 
        query: str, 
        verbose: bool, 
        bounding_box: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Optimized agent-based search with improved error handling
        """
        system_prompt = """
        You are an intelligent search agent. Use these optimized functions:
        
        1. calculate_similarity(query, column, table_name=None) - Vector similarity search
        2. find_table_by_elements(elements, column) - Find table by elements  
        3. ids_of_elements(table_name, col_type=[], col_name=[]) - Get filtered results
        
        Always store final results in 'final_id_list' variable.
        Use proper error handling and optimization strategies.
        """
        
        namespace = {
            name: obj for name, obj in globals().items() 
            if isinstance(obj, types.FunctionType)
        }
        namespace["final_id_list"] = []
        
        messages = messages_initial_template(system_prompt, query)
        max_rounds = 5  # Reduced from 10 for better performance
        
        for round_num in range(1, max_rounds + 1):
            try:
                code_result = chat_single(messages, temperature=0.3)  # Lower temperature for consistency
                messages.append(message_template('assistant', code_result))
                
                if 'python' in code_result:
                    code_return = str(
                        execute_and_display(
                            extract_python_code(code_result),
                            namespace
                        )
                    )
                else:
                    code_return = code_result
                
                messages.append(message_template('user', str(code_return)))
                
                # Check for completion
                if 'final_id_list' in namespace and namespace["final_id_list"]:
                    if 'traceback' not in str(code_return).lower():
                        return self._merge_id_lists(namespace["final_id_list"])
                
                if verbose:
                    safe_print(f"Round {round_num}: {code_result[:100]}...")
                
            except Exception as e:
                safe_print(f"Search round {round_num} error: {e}")
                continue
        
        return None
    
    @cached()
    def _merge_id_lists(self, id_list_collection: List[Dict]) -> Dict[str, Any]:
        """
        Merge ID lists with caching and optimization
        """
        if not id_list_collection:
            return {'id_list': {}, 'table_name': ''}
        
        if isinstance(id_list_collection, dict):
            return id_list_collection
        
        merged_result = {'id_list': {}, 'table_name': ''}
        
        for item in id_list_collection:
            if isinstance(item, dict):
                if 'id_list' in item:
                    merged_result['id_list'].update(item['id_list'])
                if 'table_name' in item and not merged_result['table_name']:
                    merged_result['table_name'] = item['table_name']
        
        return merged_result


# Global entity processor instance
entity_processor = EntityProcessor()


# ============================================================================
# Optimized Bounding Box Functions
# ============================================================================

class BoundingBoxManager:
    """
    Optimized bounding box management with caching and error handling
    """
    
    def __init__(self):
        self.bbox_cache: Dict[str, Dict[str, Any]] = {}
    
    @cached()
    def set_bounding_box(self, region_name: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Set bounding box with caching and optimization
        """
        if not region_name:
            session['globals_dict'] = None
            return {'geo_map': ''}
        
        try:
            # Get bounding box data
            coords, wkb_hex, response_str = find_boundbox(region_name)
            
            bounding_box_dict = {
                "bounding_box_region_name": region_name,
                "bounding_coordinates": coords,
                "bounding_wkb": wkb_hex
            }
            
            # Create geo dictionary
            geo_dict = {
                region_name: wkb.loads(bytes.fromhex(wkb_hex))
            }
            
            # Update session
            session['globals_dict'] = bounding_box_dict
            session.modified = True
            
            # Prepare return dictionary
            return_dict = {'geo_map': geo_dict}
            return_dict.update(bounding_box_dict)
            
            return return_dict
        
        except Exception as e:
            safe_print(f"Bounding box error for {region_name}: {e}")
            return {'geo_map': '', 'error': str(e)}
    
    @cached()
    def process_bounding_box_query(self, query: str) -> Optional[List[float]]:
        """
        Process bounding box query with directional modifiers
        """
        if not query:
            return None
        
        ask_prompt = """
        Adjust bounding box coordinates based on directional modifiers in the query.
        Return JSON: {"boundingbox": [coordinates]}
        """
        
        try:
            messages = [
                message_template('system', ask_prompt),
                message_template('user', str(query))
            ]
            
            result = chat_single(messages, 'json', 'gpt-4o-2024-05-13')
            result_data = json.loads(result)
            
            return result_data.get('boundingbox', [])
        
        except Exception as e:
            safe_print(f"Bounding box query processing error: {e}")
            return None


# Global bounding box manager instance
bbox_manager = BoundingBoxManager()


# ============================================================================
# Main Optimized API Functions
# ============================================================================

def pick_match(
    query_feature: str, 
    table_name: str, 
    verbose: bool = False,
    bounding_box: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Optimized main feature matching function
    
    Args:
        query_feature: Feature query string
        table_name: Target table name
        verbose: Enable verbose logging
        bounding_box: Optional bounding box filter
    
    Returns:
        Dictionary containing match results
    """
    try:
        return feature_matcher.process_feature_matching(
            query_feature, 
            table_name, 
            verbose, 
            bounding_box
        )
    except Exception as e:
        safe_print(f"Feature matching error: {e}")
        raise


def id_list_of_entity(
    query: str, 
    verbose: bool = False, 
    bounding_box: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Optimized entity ID list retrieval
    
    Args:
        query: Entity query string
        verbose: Enable verbose logging
        bounding_box: Optional bounding box filter
    
    Returns:
        Dictionary containing entity ID list
    """
    try:
        return entity_processor.process_entity_list(query, verbose, bounding_box)
    except Exception as e:
        safe_print(f"Entity processing error: {e}")
        return None


def geo_filter(
    query: str, 
    id_list_subject: Union[str, Dict], 
    id_list_object: Union[str, Dict],
    bounding_box: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Optimized geographic filtering function
    
    Args:
        query: Geographic query string
        id_list_subject: Subject entity list
        id_list_object: Object entity list
        bounding_box: Optional bounding box filter
    
    Returns:
        Dictionary containing geographic filter results
    """
    try:
        return geo_processor.process_geographic_filter(
            query, 
            id_list_subject, 
            id_list_object, 
            bounding_box
        )
    except Exception as e:
        safe_print(f"Geographic filtering error: {e}")
        return {'error': str(e)}


def set_bounding_box(region_name: str, query: Optional[str] = None) -> Dict[str, Any]:
    """
    Optimized bounding box setting function
    
    Args:
        region_name: Name of the region
        query: Optional query string
    
    Returns:
        Dictionary containing bounding box information
    """
    try:
        return bbox_manager.set_bounding_box(region_name, query)
    except Exception as e:
        safe_print(f"Bounding box setting error: {e}")
        return {'geo_map': '', 'error': str(e)}


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

# Maintain backward compatibility with existing function names
def error_test():
    """Backward compatibility function"""
    safe_print("Optimized error test function")
    raise Exception("Test exception from optimized module")


def vice_versa(query: str, messages: Optional[List] = None) -> str:
    """
    Optimized inverse query generation with caching
    """
    @cached()
    def _generate_inverse(q: str) -> str:
        ask_prompt = """
        Rewrite input to inverse the judgement. Return JSON: {"result": "inverted statement"}
        Examples:
        - "good for agriculture" -> "bad for agriculture"
        - "negative for planting" -> "positive for planting"
        """
        
        try:
            msgs = messages or []
            msgs.extend([
                message_template('system', ask_prompt),
                message_template('user', q)
            ])
            
            result = chat_single(msgs, 'json')
            if isinstance(result, str):
                json_result = json.loads(result)
                return json_result.get('result', f'not {q}')
        
        except Exception as e:
            safe_print(f"Inverse generation error: {e}")
        
        return f'not {q}'
    
    return _generate_inverse(query)


# Additional backward compatibility functions
judge_geo_relation = geo_processor.judge_geo_relation
process_boundingbox = bbox_manager.process_bounding_box_query
judge_table = entity_processor.judge_table_name

# Legacy print function
print_modify = safe_print


# ============================================================================
# Module Cleanup and Optimization
# ============================================================================

def clear_all_caches() -> None:
    """
    Clear all caches for memory management
    """
    performance_cache.clear()
    
    # Clear function-level caches
    for obj in [query_processor, similarity_calc, feature_matcher, geo_processor, entity_processor, bbox_manager]:
        if hasattr(obj, 'cache_clear'):
            obj.cache_clear()


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring
    """
    return {
        'performance_cache_size': len(performance_cache._cache),
        'global_sets_size': {
            'all_fclass_set': len(all_fclass_set),
            'all_name_set': len(all_name_set)
        },
        'dictionaries_size': {
            'fclass_dict_4_similarity': len(fclass_dict_4_similarity),
            'name_dict_4_similarity': len(name_dict_4_similarity)
        }
    }


# ============================================================================
# Module Initialization
# ============================================================================

if __name__ == "__main__":
    safe_print("Optimized Ask Functions Agent module loaded successfully")
    safe_print(f"Cache stats: {get_cache_stats()}")