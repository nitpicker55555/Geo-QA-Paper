# -*- coding: utf-8 -*-
"""
Entity Search Agent - Fast Implementation

This module provides fast entity search functionality using ChromaDB and LLM-based intent detection.
Searches in both fclass (labels) and name collections to find relevant entities.

Features:
- Direct semantic search using ChromaDB for fclass and name matching
- LLM-based query analysis to determine search intent (label/name/intersection/union)  
- Fuzzy search fallback using iterative LLM agent when no direct matches are found
- Geographic bounding box filtering support
- Multiple table search with result merging

Fuzzy Search Feature:
When direct searches for both fclass and name terms return empty results, the module
automatically triggers an iterative fuzzy search agent that:
1. Analyzes available data samples from each table
2. Uses LLM to suggest alternative search terms based on user intent
3. Iteratively refines search terms until satisfactory results are found
4. Determines the optimal search intent (label/name/intersection/union) for results
5. Provides reasoning for the search strategy used

Example usage:
    >>> result = id_list_of_entity_fast("coffee shop", verbose=True)  # Chinese for "coffee shop"
    >>> # Will trigger fuzzy search and likely find "cafe", "restaurant" matches
"""

import logging
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union

# Create a clear, one-way dependency on ask_functions_agent for helpers and data
from .ask_functions_agent import (
    calculate_similarity_chroma,
    name_cosin_list,
    find_keys_by_values,
    merge_dicts,
    judge_table,
    all_fclass_set,
    all_name_set,
    fclass_dict_4_similarity,
    name_dict_4_similarity
)

from .geo_functions import (
    ids_of_type,
    ids_of_attribute,
    col_name_mapping_dict,
    similar_ori_table_name_dict
)

from services.chat_py import general_gpt_without_memory

def _analyze_query_llm(query: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Uses an LLM to analyze the user's query and extract its semantic structure.

    This function acts as the "brain" of the search process, deconstructing a natural
    language query into a structured JSON object that dictates the rest of the
    search pipeline.

    Args:
        query (str): The original, natural language user query.
        logger (logging.Logger): Logger instance for verbose output.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed query structure.
                        Example:
                        {
                            "intent": "intersection",
                            "table_scope": "landuse",
                            "fclass_terms": ["park"],
                            "name_terms": ["English Garden"]
                        }
    """
    table_list = list(col_name_mapping_dict.keys())
    sys_prompt = f"""
    Analyze the user's natural language query for searching a geospatial database. Deconstruct the query into its primary components: intent, search terms (for categories and names), and any specified table.

    The user can query for:
    1.  An entire data table (e.g., "show me all buildings").
    2.  Items by category/class, known as `fclass` (e.g., "restaurants," "parks").
    3.  Items by a specific proper name (e.g., "Marienplatz").
    4.  A combination of name and category, indicating an "AND" condition (e.g., "the river Isar," "a park named 'English Garden'").
    5.  A combination of multiple categories or names, indicating an "OR" condition (e.g., "schools or parks," "show me greenery and buildings").

    Your task is to parse the query and return a structured JSON object.

    Query: "{query}"

    Available data tables: {table_list}
    Common aliases for tables: {similar_ori_table_name_dict}

    Respond in JSON format with the following structure:
    {{
      "intent": "whole_table" | "intersection" | "union" | "label" | "name",
      "table_scope": "the_canonical_table_name" | null,
      "fclass_terms": ["term1", "term2", ...] | [],
      "name_terms": ["term1", "term2", ...] | []
    }}

    Guidelines for JSON output:
    - "intent":
      - "whole_table": For requests for all items in a table.
      - "label": For queries about general categories only.
      - "name": For queries about specific named entities only.
      - "intersection": For "AND" logic (e.g., a name within a category).
      - "union": For "OR" logic (e.g., "schools or parks").
    - "table_scope": If the query explicitly mentions a table to search within (e.g., "... in the buildings data"), put the canonical table name here. Otherwise, null.
    - "fclass_terms": A list of extracted category/class terms. Normalize terms (e.g., "universities" -> "university"). Should be empty if not present.
    - "name_terms": A list of extracted proper name terms. Should be empty if not present.

    Examples:
    - Query: "show all buildings" -> {{"intent": "whole_table", "table_scope": "buildings", "fclass_terms": [], "name_terms": []}}
    - Query: "restaurants" -> {{"intent": "label", "table_scope": null, "fclass_terms": ["restaurant"], "name_terms": []}}
    - Query: "Marienplatz" -> {{"intent": "name", "table_scope": null, "fclass_terms": [], "name_terms": ["Marienplatz"]}}
    - Query: "park named English Garden" -> {{"intent": "intersection", "table_scope": "landuse", "fclass_terms": ["park"], "name_terms": ["English Garden"]}}
    - Query: "the Isar river" -> {{"intent": "intersection", "table_scope": "lines", "fclass_terms": ["river"], "name_terms": ["Isar"]}}
    - Query: "schools or hospitals" -> {{"intent": "union", "table_scope": null, "fclass_terms": ["school", "hospital"], "name_terms": []}}
    - Query: "show me buildings and greenery" -> {{"intent": "union", "table_scope": null, "fclass_terms": ["building", "greenery"], "name_terms": []}}
    - Query: "universities in the points table" -> {{"intent": "label", "table_scope": "points", "fclass_terms": ["university"], "name_terms": []}}
    """
    try:
        result_str = general_gpt_without_memory(sys_prompt=sys_prompt, query=query, json_mode='json', verbose=False)
        result_json = json.loads(result_str) if isinstance(result_str, str) else result_str
        
        # Validate the structure
        if isinstance(result_json, dict) and "intent" in result_json:
            logger.info(f"LLM analysis successful: {result_json}")
            # Ensure lists are present
            result_json.setdefault('fclass_terms', [])
            result_json.setdefault('name_terms', [])
            return result_json
        else:
            logger.warning(f"LLM analysis returned invalid structure: {result_json}. Falling back to default.")
            return {"intent": "label", "table_scope": None, "fclass_terms": [query], "name_terms": []}
            
    except Exception as e:
        logger.error(f"Error in LLM query analysis: {e}. Falling back to default search.")
        # Fallback to treat the whole query as a single label search term
        return {"intent": "label", "table_scope": None, "fclass_terms": [query], "name_terms": []}

def _guess_best_table(query: str, logger: logging.Logger) -> str:
    """
    Uses LLM to guess the most relevant table for the given query.
    
    Args:
        query: The search query
        logger: Logger instance
        
    Returns:
        The most likely table name
    """
    table_list = list(col_name_mapping_dict.keys())
    
    guess_prompt = f"""
    Based on the user's query, determine which database table is most likely to contain relevant results.
    
    Available tables: {table_list}
    User query: "{query}"
    
    Table descriptions:
    - buildings: Contains building data, structures, houses, etc.
    - landuse: Contains land use data like parks, forests, residential areas, etc.
    - lines: Contains linear features like roads, rivers, railways, etc.
    - points: Contains point features like shops, restaurants, amenities, etc.
    - soil: Contains soil type data
    
    Respond in JSON format with the table name:
    {{
      "table_name": "most_likely_table_name",
      "reasoning": "brief explanation of why this table was chosen"
    }}
    """
    
    try:
        result_str = general_gpt_without_memory(
            sys_prompt=guess_prompt,
            query="",
            json_mode='json',
            verbose=False
        )
        result_json = json.loads(result_str) if isinstance(result_str, str) else result_str
        
        if isinstance(result_json, dict) and "table_name" in result_json:
            guessed_table = result_json["table_name"].strip().lower()
            reasoning = result_json.get("reasoning", "")
            logger.info(f"LLM guessed table: {guessed_table} (reason: {reasoning})")
            
            # Validate the guess
            if guessed_table in table_list:
                return guessed_table
            else:
                logger.warning(f"LLM guessed invalid table '{guessed_table}', defaulting to 'points'")
                return 'points'  # Default fallback
        else:
            logger.warning(f"LLM returned invalid JSON structure: {result_json}, defaulting to 'points'")
            return 'points'  # Default fallback
            
    except Exception as e:
        logger.error(f"Error in table guessing: {e}, defaulting to 'points'")
        return 'points'  # Default fallback

def _get_table_samples(query: str, table_scope: Optional[str], bounding_box: Optional[Dict[str, Any]], 
                      logger: logging.Logger, sample_size: int = 10) -> Dict[str, Dict[str, List[str]]]:
    """
    Get sample data from the most relevant table to help the LLM understand available data.
    
    Args:
        query: The original search query
        table_scope: Specific table scope from query analysis, if any
        bounding_box: Geographic bounding box for filtering
        logger: Logger instance
        sample_size: Number of samples to get from each category
        
    Returns:
        Dict with structure: {table_name: {'fclass_samples': [...], 'name_samples': [...]}}
    """
    # Determine which table to sample from
    if table_scope:
        target_table = table_scope
        logger.info(f"Using specified table scope: {target_table}")
    else:
        target_table = _guess_best_table(query, logger)
        logger.info(f"Agent guessed table: {target_table}")
    
    samples = {target_table: {'fclass_samples': [], 'name_samples': []}}
    
    try:
        # Get fclass samples using ids_of_attribute
        fclass_list = ids_of_attribute(target_table, specific_col="fclass", bounding_box_coordinates=bounding_box)
        if fclass_list:
            fclass_samples = random.sample(fclass_list, min(sample_size, len(fclass_list)))
            samples[target_table]['fclass_samples'] = fclass_samples
            logger.info(f"Got {len(fclass_samples)} fclass samples from {target_table}")
        
        # Get name samples (skip 'soil' table as it might not have names)
        if target_table != 'soil':
            name_list = ids_of_attribute(target_table, specific_col='name', bounding_box_coordinates=bounding_box)
            if name_list:
                name_samples = random.sample(name_list, min(sample_size, len(name_list)))
                samples[target_table]['name_samples'] = name_samples
                logger.info(f"Got {len(name_samples)} name samples from {target_table}")
                
    except Exception as e:
        logger.error(f"Error getting samples from table {target_table}: {e}")
    
    return samples

def _iterative_fuzzy_search(original_query: str, table_scope: Optional[str], 
                          current_fclass_set: set[str], current_name_set: set[str],
                          current_fclass_dict: Dict[str, set[str]], current_name_dict: Dict[str, set[str]],
                          bounding_box: Optional[Dict[str, Any]], logger: logging.Logger, 
                          max_iterations: int = 3) -> Dict[str, Any]:
    """
    Iterative agent that tries to find matches by reformulating the query using LLM guidance.
    
    Args:
        original_query: The original user query
        table_scope: Specific table scope from query analysis, if any
        current_fclass_set: Set of all available fclass values
        current_name_set: Set of all available name values
        current_fclass_dict: Dictionary mapping table names to fclass sets
        current_name_dict: Dictionary mapping table names to name sets
        bounding_box: Geographic bounding box for filtering
        logger: Logger instance
        max_iterations: Maximum number of search iterations
        
    Returns:
        Dictionary with 'fclass_matches', 'name_matches', and 'final_intent'
    """
    logger.info(f"Starting iterative fuzzy search for query: '{original_query}'")
    
    # Get sample data to help LLM understand what's available
    table_samples = _get_table_samples(original_query, table_scope, bounding_box, logger, sample_size=8)
    
    # Prepare context for the LLM
    context_info = []
    for table_name, samples in table_samples.items():
        fclass_examples = samples['fclass_samples'][:5]  # Limit to avoid too long prompt
        name_examples = samples['name_samples'][:5]
        context_info.append(f"Table '{table_name}': fclass examples: {fclass_examples}, name examples: {name_examples}")
    
    context_str = "\n".join(context_info)
    
    for iteration in range(max_iterations):
        logger.info(f"Fuzzy search iteration {iteration + 1}/{max_iterations}")
        
        fuzzy_prompt = f"""
        The user's original query "{original_query}" didn't match any entries in our geospatial database directly.
        
        Available data in our database:
        {context_str}
        
        Your task is to help find relevant matches by:
        1. Analyzing what the user might be looking for
        2. Suggesting alternative search terms that might match the available data
        3. Determining the best search strategy (label/name/intersection/union)
        
        Please suggest 2-3 alternative fclass (category) terms and 2-3 alternative name terms that might match the user's intent.
        Also determine what type of search would be most appropriate.
        
        Respond in JSON format:
        {{
          "reasoning": "Brief explanation of your analysis",
          "suggested_fclass_terms": ["term1", "term2", "term3"],
          "suggested_name_terms": ["term1", "term2", "term3"],
          "recommended_intent": "label" | "name" | "intersection" | "union",
          "confidence": 0.1-1.0
        }}
        
        Original query: "{original_query}"
        """
        
        try:
            result_str = general_gpt_without_memory(
                sys_prompt=fuzzy_prompt, 
                query="", 
                json_mode='json', 
                verbose=False
            )
            suggestion = json.loads(result_str) if isinstance(result_str, str) else result_str
            
            # Validate that suggestion is a dictionary
            if not isinstance(suggestion, dict):
                logger.warning(f"LLM returned non-dict suggestion: {type(suggestion)}, skipping iteration")
                continue
            
            logger.info(f"LLM suggestion: {suggestion}")
            
            # Try the suggested terms
            all_fclass_matches = set()
            all_name_matches = []
            
            # Search with suggested fclass terms
            suggested_fclass = suggestion.get('suggested_fclass_terms', [])
            for term in suggested_fclass:
                try:
                    fclass_matches, _ = calculate_similarity_chroma(
                        query=term, 
                        give_list=current_fclass_set, 
                        mode='fclass'
                    )
                    if isinstance(fclass_matches, (list, set)):
                        all_fclass_matches.update(fclass_matches)
                        logger.info(f"Found {len(fclass_matches)} fclass matches for term '{term}'")
                except Exception as e:
                    logger.warning(f"Error searching for suggested fclass term '{term}': {e}")
            
            # Search with suggested name terms
            suggested_names = suggestion.get('suggested_name_terms', [])
            for term in suggested_names:
                try:
                    name_matches, _ = name_cosin_list(term, current_name_set)
                    all_name_matches.extend(name_matches)
                    logger.info(f"Found {len(name_matches)} name matches for term '{term}'")
                except Exception as e:
                    logger.warning(f"Error searching for suggested name term '{term}': {e}")
            
            # Check if we found anything useful
            total_matches = len(all_fclass_matches) + len(all_name_matches)
            confidence = suggestion.get('confidence', 0.5)
            
            logger.info(f"Iteration {iteration + 1}: Found {total_matches} total matches with confidence {confidence}")
            
            # If we found good matches or confidence is high enough, return results
            if total_matches > 0 and (confidence >= 0.7 or total_matches >= 3):
                logger.info(f"Accepting fuzzy search results: {total_matches} matches, confidence: {confidence}")
                return {
                    'fclass_matches': all_fclass_matches,
                    'name_matches': all_name_matches,
                    'final_intent': suggestion.get('recommended_intent', 'union'),
                    'reasoning': suggestion.get('reasoning', ''),
                    'iteration_used': iteration + 1
                }
            
            # If this is the last iteration, return what we have
            if iteration == max_iterations - 1:
                logger.info(f"Last iteration: returning best available results")
                return {
                    'fclass_matches': all_fclass_matches,
                    'name_matches': all_name_matches,
                    'final_intent': suggestion.get('recommended_intent', 'union'),
                    'reasoning': suggestion.get('reasoning', ''),
                    'iteration_used': iteration + 1
                }
                
        except Exception as e:
            logger.error(f"Error in fuzzy search iteration {iteration + 1}: {e}")
            continue
    
    # If all iterations failed, return empty results
    logger.warning("All fuzzy search iterations failed")
    return {
        'fclass_matches': set(),
        'name_matches': [],
        'final_intent': 'label',
        'reasoning': 'Fuzzy search failed to find matches',
        'iteration_used': max_iterations
    }

def id_list_of_entity_fast(query: str, verbose: bool = True, bounding_box: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced entity search function that:
    1. Searches in both fclass (labels) and name collections in chromadb
    2. Uses LLM to determine if query represents label, name, or intersection
    3. Returns results using ids_of_type function
    
    Args:
        query (str): Search query string
        verbose (bool): Enable verbose logging
        bounding_box (Optional[Dict]): Geographic bounding box for filtering
        
    Returns:
        Dict[str, Any]: Results from ids_of_type function with structure:
            {
                'id_list': {id1: data1, id2: data2, ...},
                'geo_map': {...}
            }
            
    Raises:
        ValueError: If query is empty or invalid
        Exception: If search operations fail
        
    Example:
        >>> # The module must be imported where dependencies are already loaded
        >>> result = id_list_of_entity_fast("restaurants", verbose=True)
        >>> result = id_list_of_entity_fast("Isar river", verbose=True)
        >>> result = id_list_of_entity_fast("Main Street", verbose=True)
    """
    
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
    
    logger.info(f"Starting entity search for query: '{query}'")
    
    # Input validation and preprocessing
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    
    # Preprocess query
    query_original = query
    query = query.lower().strip()
    query = query.replace("strasse", 'straße')
    logger.info(f"Preprocessed query: '{query}'")
    
    # Handle bounding box - no session dependency
    if bounding_box is None:
        logger.info("No bounding_box provided, using global data")
    else:
        logger.info(f"Using provided bounding_box: {bounding_box}")
    
    # Prepare search data sets first
    logger.info("Preparing search data sets for exact matching...")
    tables_to_process = col_name_mapping_dict.keys()
    
    current_fclass_set = set()
    current_name_set = set()
    current_fclass_dict = {}
    current_name_dict = {}

    if bounding_box is not None:
        logger.info(f"Building data sets for tables with bounding box")
        for table_name in tables_to_process:
            fclass_set = ids_of_attribute(table_name, specific_col="fclass", bounding_box_coordinates=bounding_box)
            current_fclass_dict[table_name] = fclass_set
            current_fclass_set.update(fclass_set)
            
            if table_name != 'soil':
                name_set = ids_of_attribute(table_name, specific_col='name', bounding_box_coordinates=bounding_box)
                current_name_dict[table_name] = name_set
                current_name_set.update(name_set)
    else:
        logger.info("Using globally pre-loaded data sets.")
        current_fclass_set = all_fclass_set
        current_name_set = all_name_set
        current_fclass_dict = fclass_dict_4_similarity
        current_name_dict = name_dict_4_similarity
    
    # Step -1: Try exact matching first (before LLM analysis)
    logger.info(f"Step -1: Attempting exact match for query: '{query}'")
    exact_fclass_matches = set()
    exact_name_matches = []
    
    # Check for exact match in fclass
    if query in current_fclass_set:
        exact_fclass_matches.add(query)
        logger.info(f"Found exact fclass match: '{query}'")
    
    # Check for exact match in name
    if query in current_name_set:
        exact_name_matches.append(query)
        logger.info(f"Found exact name match: '{query}'")
    
    # Also check the original query (with original case)
    if query_original in current_fclass_set:
        exact_fclass_matches.add(query_original)
        logger.info(f"Found exact fclass match (original case): '{query_original}'")
    
    if query_original in current_name_set:
        exact_name_matches.append(query_original)
        logger.info(f"Found exact name match (original case): '{query_original}'")
    
    # If we found exact matches, use them directly without LLM analysis
    if exact_fclass_matches or exact_name_matches:
        logger.info(f"Exact matches found! Skipping LLM analysis. fclass: {exact_fclass_matches}, name: {exact_name_matches}")
        
        # Find relevant tables for exact matches
        table_fclass_dicts = find_keys_by_values(current_fclass_dict, exact_fclass_matches)
        table_name_dicts = find_keys_by_values(current_name_dict, exact_name_matches)
        
        all_id_lists = []
        
        # Process fclass exact matches
        for table_name in table_fclass_dicts.keys():
            fclass_list = table_fclass_dicts[table_name]
            logger.info(f"Table {table_name}: Getting exact fclass results for {len(fclass_list)} items.")
            each_id_list = ids_of_type(table_name, {
                'non_area_col': {'fclass': set(fclass_list), 'name': set()},
                'area_num': None
            }, bounding_box=bounding_box)
            all_id_lists.append(each_id_list)
        
        # Process name exact matches
        for table_name in table_name_dicts.keys():
            name_list = table_name_dicts[table_name]
            logger.info(f"Table {table_name}: Getting exact name results for {len(name_list)} items.")
            each_id_list = ids_of_type(table_name, {
                'non_area_col': {'fclass': set(), 'name': set(name_list)},
                'area_num': None
            }, bounding_box=bounding_box)
            all_id_lists.append(each_id_list)
        
        # Merge and return results
        if all_id_lists:
            merged_result = merge_dicts(all_id_lists)
            final_count = len(merged_result.get('id_list', {}))
            logger.info(f"Exact match completed. Returning {final_count} results.")
            return merged_result
    
    # Step 0: If no exact match, proceed with LLM analysis
    logger.info("No exact matches found. Proceeding with LLM analysis...")
    logger.info("Step 0: Analyzing query structure with LLM")
    analysis = _analyze_query_llm(query_original, logger)
    logger.info(f"LLM analysis result: {analysis}")

    query_intent = analysis.get("intent")
    table_scope = analysis.get("table_scope")
    fclass_terms = analysis.get("fclass_terms", [])
    name_terms = analysis.get("name_terms", [])

    # Handle whole_table intent directly
    if query_intent == 'whole_table' and table_scope:
        logger.info(f"Executing whole-table search for table: {table_scope}")
        return ids_of_type(table_scope, {
            'non_area_col': {'fclass': set(), 'name': set()},
            'area_num': None
        }, bounding_box=bounding_box)

    # If table_scope is specified, we might need to update the data sets
    if table_scope and table_scope not in current_fclass_dict:
        logger.info(f"Table scope '{table_scope}' specified but not in current data sets. Updating...")
        if bounding_box is not None:
            fclass_set = ids_of_attribute(table_scope, specific_col="fclass", bounding_box_coordinates=bounding_box)
            current_fclass_dict[table_scope] = fclass_set
            current_fclass_set.update(fclass_set)
            
            if table_scope != 'soil':
                name_set = ids_of_attribute(table_scope, specific_col='name', bounding_box_coordinates=bounding_box)
                current_name_dict[table_scope] = name_set
                current_name_set.update(name_set)
    
    # Step 1 & 2: Perform similarity search based on extracted terms
    all_fclass_matches = set()
    all_name_matches = []

    if fclass_terms:
        logger.info(f"Step 1: Searching for fclass terms: {fclass_terms}")
        for term in fclass_terms:
            try:
                fclass_matches, _ = calculate_similarity_chroma(
                    query=term, 
                    give_list=current_fclass_set, 
                    mode='fclass'
                )
                if isinstance(fclass_matches, (list, set)):
                    all_fclass_matches.update(fclass_matches)
                else:
                    logger.warning(f"Expected list or set from fclass search for term '{term}', but got {type(fclass_matches)}")
            except Exception as e:
                logger.warning(f"Error searching for fclass term '{term}': {e}")
    
    if name_terms:
        logger.info(f"Step 2: Searching for name terms: {name_terms}")
        for term in name_terms:
            try:
                name_matches, _ = name_cosin_list(term, current_name_set)
                all_name_matches.extend(name_matches)
            except Exception as e:
                logger.warning(f"Error searching for name term '{term}': {e}")
    
    logger.info(f"Total fclass matches: {len(all_fclass_matches)}, Total name matches: {len(all_name_matches)}")

    # Step 3: Fuzzy search fallback if no direct matches found
    if not all_fclass_matches and not all_name_matches:
        logger.info("No direct matches found, triggering iterative fuzzy search...")
        fuzzy_result = _iterative_fuzzy_search(
            original_query=query_original,
            table_scope=table_scope,
            current_fclass_set=current_fclass_set,
            current_name_set=current_name_set,
            current_fclass_dict=current_fclass_dict,
            current_name_dict=current_name_dict,
            bounding_box=bounding_box,
            logger=logger,
            max_iterations=3
        )
        
        # Use fuzzy search results
        all_fclass_matches = fuzzy_result.get('fclass_matches', set())
        all_name_matches = fuzzy_result.get('name_matches', [])
        
        # Override the original intent with the one determined by the fuzzy search agent
        query_intent = fuzzy_result.get('final_intent', query_intent)
        
        logger.info(f"Fuzzy search completed in {fuzzy_result.get('iteration_used', 0)} iterations")
        logger.info(f"Fuzzy search reasoning: {fuzzy_result.get('reasoning', 'N/A')}")
        logger.info(f"Updated intent: {query_intent}")
        logger.info(f"Fuzzy fclass matches: {len(all_fclass_matches)}, Fuzzy name matches: {len(all_name_matches)}")

    # Step 4: Find relevant tables for each match type
    table_fclass_dicts = find_keys_by_values(current_fclass_dict, all_fclass_matches)
    table_name_dicts = find_keys_by_values(current_name_dict, all_name_matches)
    
    logger.info(f"Tables for fclass matches: {list(table_fclass_dicts.keys())}")
    logger.info(f"Tables for name matches: {list(table_name_dicts.keys())}")
    
    # Step 5 & 6: Determine final tables and collect results
    all_id_lists = []
    tables_searched = set()
    
    if query_intent == 'union':
        union_tables = set(table_fclass_dicts.keys()) | set(table_name_dicts.keys())
        tables_searched = union_tables
        logger.info(f"Processing UNION search for tables: {list(union_tables)}")
        for table_name in union_tables:
            # Get fclass results if any
            if table_name in table_fclass_dicts:
                fclass_list = table_fclass_dicts[table_name]
                logger.info(f"Table {table_name}: Getting UNION fclass results for {len(fclass_list)} items.")
                each_id_list = ids_of_type(table_name, {'non_area_col': {'fclass': set(fclass_list), 'name': set()}, 'area_num': None}, bounding_box=bounding_box)
                all_id_lists.append(each_id_list)

            # Get name results if any
            if table_name in table_name_dicts:
                name_list = table_name_dicts[table_name]
                logger.info(f"Table {table_name}: Getting UNION name results for {len(name_list)} items.")
                each_id_list = ids_of_type(table_name, {'non_area_col': {'fclass': set(), 'name': set(name_list)}, 'area_num': None}, bounding_box=bounding_box)
                all_id_lists.append(each_id_list)
    else:
        # Logic for 'intersection', 'label', 'name'
        active_tables = set()
        if query_intent == 'intersection':
            active_tables = set(table_fclass_dicts.keys()) & set(table_name_dicts.keys())
            if not active_tables:
                active_tables = set(table_fclass_dicts.keys()) | set(table_name_dicts.keys())
        elif query_intent == 'label':
            active_tables = set(table_fclass_dicts.keys())
        elif query_intent == 'name':
            active_tables = set(table_name_dicts.keys())
        
        tables_searched = active_tables
        logger.info(f"Processing '{query_intent}' search for tables: {list(active_tables)}")
        for table_name in active_tables:
            # Determine what to search for in this table
            fclass_list = []
            name_list = []
            
            if query_intent in ['label', 'intersection'] and table_name in table_fclass_dicts:
                fclass_list = table_fclass_dicts[table_name]
            
            if query_intent in ['name', 'intersection'] and table_name in table_name_dicts:
                name_list = table_name_dicts[table_name]
            
            # Special handling for intersection mode
            if query_intent == 'intersection':
                # For intersection, we need both fclass and name matches
                if table_name in table_fclass_dicts and table_name in table_name_dicts:
                    fclass_list = table_fclass_dicts[table_name]
                    name_list = table_name_dicts[table_name]
                elif table_name in table_fclass_dicts:
                    fclass_list = table_fclass_dicts[table_name]
                    name_list = []
                elif table_name in table_name_dicts:
                    fclass_list = []
                    name_list = table_name_dicts[table_name]
            
            logger.info(f"Table {table_name}: fclass_list={len(fclass_list)}, name_list={len(name_list)}")
            
            # Call ids_of_type with the determined parameters
            try:
                each_id_list = ids_of_type(table_name, {
                    'non_area_col': {
                        'fclass': set(fclass_list), 
                        'name': set(name_list)
                    },
                    'area_num': None
                }, bounding_box=bounding_box)
                
                all_id_lists.append(each_id_list)
                logger.info(f"Retrieved {len(each_id_list.get('id_list', {}))} items from table {table_name}")
                
            except Exception as e:
                logger.error(f"Error retrieving data from table {table_name}: {e}")
    
    # Step 7: Merge results
    if not all_id_lists:
        logger.warning("No results found for any table")
        return {'id_list': {}, 'geo_map': {}}
    
    merged_result = merge_dicts(all_id_lists)
    final_count = len(merged_result.get('id_list', {}))
    logger.info(f"Final merged result: {final_count} total items")
    
    # Step 8: Additional fuzzy search fallback if final count is 0
    if final_count == 0:
        logger.info("Final count is 0, triggering additional fuzzy search...")
        fuzzy_result = _iterative_fuzzy_search(
            original_query=query_original,
            table_scope=table_scope,
            current_fclass_set=current_fclass_set,
            current_name_set=current_name_set,
            current_fclass_dict=current_fclass_dict,
            current_name_dict=current_name_dict,
            bounding_box=bounding_box,
            logger=logger,
            max_iterations=3
        )
        
        # Use fuzzy search results if we got any
        if fuzzy_result.get('fclass_matches') or fuzzy_result.get('name_matches'):
            fuzzy_fclass_matches = fuzzy_result.get('fclass_matches', set())
            fuzzy_name_matches = fuzzy_result.get('name_matches', [])
            fuzzy_intent = fuzzy_result.get('final_intent', 'union')
            
            logger.info(f"Additional fuzzy search found {len(fuzzy_fclass_matches)} fclass and {len(fuzzy_name_matches)} name matches")
            
            # Find relevant tables for fuzzy matches
            table_fclass_dicts = find_keys_by_values(current_fclass_dict, fuzzy_fclass_matches)
            table_name_dicts = find_keys_by_values(current_name_dict, fuzzy_name_matches)
            
            # Process fuzzy results with the same logic as before
            fuzzy_id_lists = []
            if fuzzy_intent == 'union':
                union_tables = set(table_fclass_dicts.keys()) | set(table_name_dicts.keys())
                for table_name in union_tables:
                    if table_name in table_fclass_dicts:
                        fclass_list = table_fclass_dicts[table_name]
                        each_id_list = ids_of_type(table_name, {'non_area_col': {'fclass': set(fclass_list), 'name': set()}, 'area_num': None}, bounding_box=bounding_box)
                        fuzzy_id_lists.append(each_id_list)
                    if table_name in table_name_dicts:
                        name_list = table_name_dicts[table_name]
                        each_id_list = ids_of_type(table_name, {'non_area_col': {'fclass': set(), 'name': set(name_list)}, 'area_num': None}, bounding_box=bounding_box)
                        fuzzy_id_lists.append(each_id_list)
            else:
                # Handle other intents
                active_tables = set()
                if fuzzy_intent == 'intersection':
                    active_tables = set(table_fclass_dicts.keys()) & set(table_name_dicts.keys())
                    if not active_tables:
                        active_tables = set(table_fclass_dicts.keys()) | set(table_name_dicts.keys())
                elif fuzzy_intent == 'label':
                    active_tables = set(table_fclass_dicts.keys())
                elif fuzzy_intent == 'name':
                    active_tables = set(table_name_dicts.keys())
                
                for table_name in active_tables:
                    fclass_list = table_fclass_dicts.get(table_name, [])
                    name_list = table_name_dicts.get(table_name, [])
                    
                    each_id_list = ids_of_type(table_name, {
                        'non_area_col': {'fclass': set(fclass_list), 'name': set(name_list)},
                        'area_num': None
                    }, bounding_box=bounding_box)
                    fuzzy_id_lists.append(each_id_list)
            
            # Merge fuzzy results
            if fuzzy_id_lists:
                merged_result = merge_dicts(fuzzy_id_lists)
                final_count = len(merged_result.get('id_list', {}))
                logger.info(f"Additional fuzzy search produced {final_count} results")
    
    if verbose:
        logger.info(f"Query processing completed successfully")
        logger.info(f"Original query: '{query_original}'")
        logger.info(f"Intent: {query_intent}")
        logger.info(f"Tables searched: {list(tables_searched)}")
        logger.info(f"Final result count: {final_count}")
    
    return merged_result
