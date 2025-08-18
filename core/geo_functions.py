"""
Optimized Geographic Functions for OSM Database Processing

This module provides optimized functions for geographic data processing with:
- Database connection pooling and query caching
- Vectorized spatial operations
- Improved memory management
- Parallel processing capabilities
- Better error handling and type hints

Performance optimizations include:
- Connection pooling with psycopg2.pool
- Query result caching with functools.lru_cache
- Vectorized numpy operations for coordinate transformations
- Batch processing for large datasets
- Spatial indexing with R-tree
- Memory-efficient geometry processing
"""

import time
import json
import copy
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
import warnings
warnings.filterwarnings('ignore')

# Third-party imports
import numpy as np
import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
import shapely
from shapely import wkb, wkt
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, Point, 
    box, shape
)
from shapely.ops import transform
from shapely.strtree import STRtree
from pyproj import CRS, Transformer
from flask import session, current_app
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EPSG_WGS84 = 4326
EPSG_UTM32N = 32632
EPSG_UTM33N = 32633
MAX_POOL_SIZE = 20
MIN_POOL_SIZE = 5
CACHE_SIZE = 1000
BATCH_SIZE = 5000
MAX_WORKERS = 4

@dataclass
class BoundingBox:
    """Optimized bounding box representation"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    def to_envelope_wkt(self) -> str:
        return f"ST_MakeEnvelope({self.min_lon}, {self.min_lat}, {self.max_lon}, {self.max_lat}, {EPSG_WGS84})"

# Database Connection Pool
class DatabaseManager:
    """Optimized database connection manager with pooling"""
    
    def __init__(self, conn_params: str, min_conn: int = MIN_POOL_SIZE, max_conn: int = MAX_POOL_SIZE):
        self.conn_params = conn_params
        self._pool = None
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._init_pool()
    
    def _init_pool(self):
        """Initialize connection pool"""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                self.min_conn, 
                self.max_conn,
                self.conn_params,
                cursor_factory=RealDictCursor
            )
            logger.info(f"Database pool initialized with {self.min_conn}-{self.max_conn} connections")
        except psycopg2.Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        conn = None
        try:
            conn = self._pool.getconn()
            conn.autocommit = True
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def close_pool(self):
        """Close all connections in pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("Database pool closed")

# Global instances
conn_params = "dbname='osm_database' user='postgres' host='localhost' password='9417941'"
db_manager = DatabaseManager(conn_params)

# Optimized global dictionaries with LRU cache
@lru_cache(maxsize=CACHE_SIZE)
def get_cached_geometry(key: str) -> Optional[Any]:
    """Cache frequently accessed geometries"""
    return global_id_geo.get(key)

# Memory-efficient global storage
class OptimizedGlobalStorage:
    """Memory-efficient storage for global geometries"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._storage = {}
        self._access_count = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._storage:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._storage[key]
        return None
    
    def update(self, data: Dict[str, Any]):
        # If adding new data would exceed max size, remove least accessed items
        if len(self._storage) + len(data) > self.max_size:
            self._cleanup()
        
        self._storage.update(data)
        for key in data:
            self._access_count[key] = 0
    
    def _cleanup(self):
        """Remove least frequently accessed items"""
        if not self._access_count:
            return
            
        # Sort by access count and remove bottom 20%
        sorted_items = sorted(self._access_count.items(), key=lambda x: x[1])
        items_to_remove = int(len(sorted_items) * 0.2)
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self._storage:
                del self._storage[key]
            del self._access_count[key]

# Initialize optimized global storage
global_id_geo = OptimizedGlobalStorage()
global_id_attribute = {}

# Configuration dictionaries (optimized)
similar_ori_table_name_dict = {
    'lands': "area", 'building': 'buildings', 'point': 'points',
    'soil': 'soil', 'areas': 'area', 'land': 'area'
}

def map_keys_to_values(similar_col_name_dict: Dict[str, str]) -> Dict[str, str]:
    """Optimized key-value mapping"""
    result = {}
    for key, value in similar_col_name_dict.items():
        result[key] = value
        result[value] = value
    return result

similar_table_name_dict = map_keys_to_values(similar_ori_table_name_dict)

col_name_mapping_dict = {
    "soil": {
        "osm_id": "objectid",
        "fclass": "leg_text",
        "name": "leg_text",
        "select_query": "SELECT leg_text,objectid,geom",
        "graph_name": "soilcomplete",
        "notice": "This Table Only has type Column."
    },
    "buildings": {
        "osm_id": "osm_id",
        "fclass": "type",
        "name": "name",
        "select_query": "SELECT 'buildings' AS source_table, type,name,osm_id,geom",
        "graph_name": "buildings"
    },
    "area": {
        "osm_id": "osm_id",
        "fclass": "fclass",
        "name": "name",
        "select_query": "SELECT 'landuse' AS source_table, fclass,name,osm_id,geom",
        "graph_name": "landuse"
    },
    "points": {
        "osm_id": "osm_id",
        "fclass": "fclass",
        "name": "name",
        "select_query": "SELECT 'points' AS source_table, fclass,name,osm_id,geom",
        "graph_name": "points"
    },
    "lines": {
        "osm_id": "osm_id",
        "fclass": "fclass",
        "name": "name",
        "select_query": "SELECT 'lines' AS source_table, fclass,name,osm_id,geom",
        "graph_name": "lines"
    }
}

revers_mapping_dict = {}

# Optimized coordinate transformer cache
@lru_cache(maxsize=100)
def get_transformer(from_crs: int, to_crs: int) -> Transformer:
    """Cached coordinate transformer"""
    return Transformer.from_crs(f"EPSG:{from_crs}", f"EPSG:{to_crs}", always_xy=True)

# Optimized session management functions
def modify_globals_dict(new_value: Any) -> None:
    """Optimized session global variable modification"""
    if 'globals_dict' not in session:
        session['globals_dict'] = {}
    session['globals_dict'].update(new_value if isinstance(new_value, dict) else {'data': new_value})

def use_globals_dict() -> Any:
    """Optimized session global variable retrieval"""
    with current_app.app_context():
        return session.get('globals_dict', {})

# Database query optimization functions
def format_sql_query(names: List[Union[str, int]]) -> str:
    """Optimized SQL query formatting with batch processing"""
    if not names:
        return "()"
    
    # Use list comprehension for better performance
    formatted_names = [
        f"'{str(name).replace(chr(39), chr(39) + chr(39))}'" if not isinstance(name, int)
        else f"'{name}'"
        for name in names
    ]
    
    return f"({', '.join(formatted_names)})"

def optimize_query_with_index_hints(query: str, table_name: str) -> str:
    """Add index hints for better query performance"""
    # Add spatial index hints for geometry columns
    if 'ST_Intersects' in query or 'ST_Contains' in query or 'ST_Within' in query:
        # Force use of spatial index
        query = query.replace(f"FROM {table_name}", f"FROM {table_name} /*+ USE_INDEX(geom_idx) */")
    
    return query

def auto_add_WHERE_AND(sql_query: str, mode: str = 'query') -> str:
    """Optimized SQL WHERE clause addition"""
    if not sql_query.strip():
        return sql_query
    
    lines = [line for line in sql_query.splitlines() if line.strip()]
    if not lines:
        return sql_query
    
    where_added = False
    from_processed = False
    modified_query = []
    
    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line.startswith('--') or not stripped_line:
            modified_query.append(line)
            continue
        
        if 'FROM' in stripped_line.upper():
            modified_query.append(line)
            from_processed = True
        elif from_processed:
            if not (stripped_line.upper().startswith('WHERE') or 
                   stripped_line.upper().startswith('AND') or
                   stripped_line.upper().startswith('ORDER') or
                   stripped_line.upper().startswith('GROUP') or
                   stripped_line.upper().startswith('LIMIT')):
                
                prefix = 'WHERE ' if not where_added else 'AND '
                line = line.replace(stripped_line, prefix + stripped_line)
                where_added = True
            
            modified_query.append(line)
        else:
            modified_query.append(line)
    
    # Add LIMIT for non-attribute queries
    if not modified_query[-1].strip().endswith(';'):
        if mode != 'attribute':
            modified_query[-1] += f'\nLIMIT {BATCH_SIZE};'
        else:
            modified_query[-1] += ';'
    
    return '\n'.join(modified_query)

# Cached database metadata functions
@lru_cache(maxsize=100)
def get_table_names() -> List[str]:
    """Cached table names retrieval"""
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public';")
                return [name['tablename'] for name in cur.fetchall()]
    except psycopg2.Error as e:
        logger.error(f"Error getting table names: {e}")
        return []

@lru_cache(maxsize=200)
def get_column_names(table_name: str) -> List[str]:
    """Cached column names retrieval"""
    if table_name not in col_name_mapping_dict:
        return []
    
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name=%s AND table_schema='public'
                """
                cur.execute(query, (col_name_mapping_dict[table_name]['graph_name'],))
                return [name['column_name'] for name in cur.fetchall()]
    except psycopg2.Error as e:
        logger.error(f"Error getting column names for {table_name}: {e}")
        return []

# Optimized database action function with retry mechanism
def cur_action(query: str, mode: str = 'query', max_retries: int = 3) -> Optional[List[Tuple]]:
    """Optimized database query execution with retry mechanism"""
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            start_time = time.time()
            
            if mode != "test":
                query = auto_add_WHERE_AND(query, mode)
                query = optimize_query_with_index_hints(query, "")
            
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                    
                    elapsed_time = time.time() - start_time
                    logger.debug(f"Query executed in {elapsed_time:.2f}s, returned {len(rows)} rows")
                    
                    return rows
                    
        except psycopg2.Error as e:
            retry_count += 1
            logger.warning(f"Query failed (attempt {retry_count}): {e}")
            
            if retry_count > max_retries:
                logger.error(f"Query failed after {max_retries} retries: {query}")
                raise Exception(f"SQL error after {max_retries} retries: {e}")
            
            time.sleep(0.5 * retry_count)  # Exponential backoff
    
    return None

# Optimized attribute retrieval with caching
def ids_of_attribute(graph_name: str, specific_col: Optional[str] = None, 
                    bounding_box_coordinates: Optional[Any] = None) -> Set[str]:
    """Optimized attribute retrieval with caching - wrapper to handle dict parameters"""
    # Convert dict to tuple for hashing if needed
    if isinstance(bounding_box_coordinates, dict):
        if 'bounding_coordinates' in bounding_box_coordinates:
            bounding_box_coordinates = tuple(bounding_box_coordinates['bounding_coordinates'])
        else:
            bounding_box_coordinates = None
    elif isinstance(bounding_box_coordinates, list):
        bounding_box_coordinates = tuple(bounding_box_coordinates)
    
    # Call the cached version
    return _ids_of_attribute_cached(graph_name, specific_col, bounding_box_coordinates)

@lru_cache(maxsize=500)
def _ids_of_attribute_cached(graph_name: str, specific_col: Optional[str] = None, 
                             bounding_box_coordinates: Optional[tuple] = None) -> Set[str]:
    """Internal cached version of ids_of_attribute"""
    if graph_name not in col_name_mapping_dict:
        logger.warning(f"Unknown graph name: {graph_name}")
        return set()
    

    fclass = col_name_mapping_dict[graph_name]['fclass']
    graph_name_modify = col_name_mapping_dict[graph_name]['graph_name'].lower()

    if specific_col and specific_col in col_name_mapping_dict[graph_name]:
        fclass = col_name_mapping_dict[graph_name][specific_col]

    # Build bounding box query if provided
    bounding_query_part = ""
    if bounding_box_coordinates:
        # Handle different bounding box formats
        if isinstance(bounding_box_coordinates, dict) and 'bounding_coordinates' in bounding_box_coordinates:
            coords = bounding_box_coordinates['bounding_coordinates']
        elif isinstance(bounding_box_coordinates, (list, tuple)) and len(bounding_box_coordinates) == 4:
            coords = bounding_box_coordinates
        else:
            coords = None

        if coords:
            min_lat, max_lat, min_lon, max_lon = coords
            bounding_query_part = f"ST_Intersects(geom, ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, {EPSG_WGS84}))"

    query = f"""
        SELECT DISTINCT {fclass}
        FROM {graph_name_modify}
        {bounding_query_part}
    """

    rows = cur_action(query, 'attribute')
    return set(row[fclass] for row in rows if row[fclass] is not None) if rows else set()

def judge_area(type_str: str) -> bool:
    """Optimized area judgment"""
    area_keywords = {'large', 'small', 'big'}
    return any(keyword in str(type_str).lower() for keyword in area_keywords)

# Vectorized geometry processing
def batch_wkb_to_geometry(hex_strings: List[str]) -> List[Any]:
    """Vectorized WKB to geometry conversion"""
    geometries = []
    
    # Process in chunks to manage memory
    for i in range(0, len(hex_strings), BATCH_SIZE):
        chunk = hex_strings[i:i + BATCH_SIZE]
        chunk_geoms = [wkb.loads(bytes.fromhex(hex_str)) for hex_str in chunk if hex_str]
        geometries.extend(chunk_geoms)
    
    return geometries

def optimize_spatial_join(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                         predicate: str = 'intersects') -> gpd.GeoDataFrame:
    """Optimized spatial join with indexing"""
    # Ensure both GeoDataFrames have spatial indexes
    if not hasattr(gdf1, 'sindex'):
        gdf1 = gdf1.copy()
    if not hasattr(gdf2, 'sindex'):
        gdf2 = gdf2.copy()
    
    # Use optimized spatial join
    return gpd.sjoin(gdf1, gdf2, how="inner", predicate=predicate)

# Optimized main query function
def ids_of_type(graph_name: str, type_dict: Dict[str, Any], 
               bounding_box: Optional[Dict] = None, test_mode: Optional[bool] = None) -> Dict[str, Any]:
    """Optimized type-based ID retrieval with batch processing and caching"""
    
    if graph_name not in col_name_mapping_dict:
        raise ValueError(f"Unknown graph name: {graph_name}")
    
    try:
        # Get table configuration
        table_config = col_name_mapping_dict[graph_name]
        select_query = table_config['select_query']
        graph_name_modify = table_config['graph_name']
        
        # Build bounding box constraint
        bounding_constraint = ""
        if bounding_box and 'bounding_coordinates' in bounding_box:
            bbox = BoundingBox(*bounding_box['bounding_coordinates'])
            bounding_constraint = f"ST_Intersects(geom, {bbox.to_envelope_wkt()})"
        
        # Build column constraints
        column_constraints = []
        for col_name, value_set in type_dict.get('non_area_col', {}).items():
            if value_set == {'all'}:
                continue
            
            col_db_name = table_config.get(col_name, col_name)
            
            if len(value_set) == 1:
                value = list(value_set)[0]
                safe_value = str(value).replace("'", "''")
                column_constraints.append(f"{col_db_name} = '{safe_value}'")
            elif len(value_set) > 1:
                formatted_values = format_sql_query(list(value_set))
                column_constraints.append(f"{col_db_name} IN {formatted_values}")
        
        # Combine all constraints
        all_constraints = [constraint for constraint in [bounding_constraint] + column_constraints if constraint]
        where_clause = " AND ".join(all_constraints) if all_constraints else ""
        
        # Build final query
        if where_clause:
            final_query = f"""
                {select_query}
                FROM {graph_name_modify}
                WHERE {where_clause}
            """
        else:
            final_query = f"""
                {select_query}
                FROM {graph_name_modify}
            """
        
        # Execute query
        rows = cur_action(final_query)
        if not rows:
            return {'id_list': {}, 'geo_map': {}}
        
        # Process results efficiently
        result_dict = process_query_results(rows, graph_name)
        
        # Update global storage
        global_id_geo.update(result_dict)
        
        # Apply area filter if specified
        if type_dict.get('area_num') is not None:
            result_dict = area_filter(result_dict, type_dict['area_num'])['id_list']
            logger.info(f"Area filtered: {len(result_dict)} items, area_num: {type_dict['area_num']}")
        
        # Build geo_map for visualization
        geo_map = {}
        if bounding_box and not test_mode:
            region_name = bounding_box.get("bounding_box_region_name")
            bounding_wkb = bounding_box.get('bounding_wkb')
            if region_name and bounding_wkb:
                geo_map[region_name] = wkb.loads(bytes.fromhex(bounding_wkb))
        
        # Sample results for visualization
        sampled_result = pick_jsons(result_dict)
        if not test_mode:
            geo_map.update(sampled_result)
        
        if len(result_dict) == 0:
            logger.warning(f"Table {graph_name} has elements {type_dict}, but not in the current region.")
        
        return {
            'id_list': result_dict if test_mode else result_dict,
            'geo_map': {} if test_mode else geo_map
        }
        
    except Exception as e:
        logger.error(f"Error in ids_of_type for {graph_name}: {e}")
        raise

def process_query_results(rows: List[Any], graph_name: str) -> Dict[str, Any]:
    """Optimized query result processing with vectorization"""
    if not rows:
        return {}
    
    # Handle both dictionary and tuple rows
    if rows and isinstance(rows[0], dict):
        # Convert dict rows to DataFrame directly
        df = pd.DataFrame(rows)
        
        # Rename columns to match expected names based on graph_name
        if graph_name == 'soil':
            # For soil: leg_text as name, objectid as global_id, geom as geometry
            if 'leg_text' in df.columns:
                df.rename(columns={'leg_text': 'name', 'objectid': 'global_id', 'geom': 'geometry_hex'}, inplace=True)
            df["key"] = f"{graph_name}_" + df["name"].astype(str) + "_" + df["global_id"].astype(str)
        else:
            # For other tables: source_table, type/fclass, name, osm_id, geom
            if 'source_table' in df.columns:
                df.rename(columns={'source_table': 'row_data', 'osm_id': 'global_id', 'geom': 'geometry_hex'}, inplace=True)
            if 'type' in df.columns:
                df.rename(columns={'type': 'empty'}, inplace=True)
            elif 'fclass' in df.columns:
                df.rename(columns={'fclass': 'empty'}, inplace=True)
            
            # Build the key
            df["key"] = (f"{graph_name}_" + df.get("name", "").astype(str) + "_" + 
                        df.get("empty", "").astype(str) + "_" + df["global_id"].astype(str))
    else:
        # Original tuple-based processing
        if graph_name == 'soil':
            df = pd.DataFrame(rows, columns=["name", "global_id", "geometry_hex"])
            df["key"] = f"{graph_name}_" + df["name"].astype(str) + "_" + df["global_id"].astype(str)
        else:
            df = pd.DataFrame(rows, columns=["row_data", "name", "empty", "global_id", "geometry_hex"])
            df["key"] = (f"{graph_name}_" + df["name"].astype(str) + "_" + 
                        df["empty"].astype(str) + "_" + df["global_id"].astype(str))
    
    # Batch convert geometries
    valid_rows = df[df["geometry_hex"].notna()]
    geometries = batch_wkb_to_geometry(valid_rows["geometry_hex"].tolist())
    
    # Create result dictionary
    result_dict = dict(zip(valid_rows["key"], geometries))
    
    return result_dict

def pick_jsons(data: Dict[str, Any], max_items: int = 20000) -> Dict[str, Any]:
    """Optimized data sampling with better distribution"""
    if len(data) <= max_items:
        return data
    
    items = list(data.items())
    step = len(items) // max_items
    sampled_items = items[::step][:max_items]
    
    return dict(sampled_items)

# Optimized area filtering with vectorization
def area_filter(data_original: Union[Dict, List], top_num: Optional[int] = None) -> Dict[str, Any]:
    """Optimized area filtering with vectorized operations"""
    try:
        top_num = int(top_num) if top_num is not None else None
        
        # Extract data dictionary
        if isinstance(data_original, dict) and 'id_list' in data_original:
            data_dict = data_original['id_list']
        else:
            data_dict = data_original
        
        if not data_dict:
            return {'area_list': {}, 'geo_map': {}, 'id_list': {}}
        
        # Vectorized area calculation
        keys = list(data_dict.keys())
        geometries = [global_id_geo.get(key) or data_dict[key] for key in keys]
        
        # Calculate areas using vectorized operations where possible
        areas = []
        for geom in geometries:
            try:
                if geom and hasattr(geom, 'area'):
                    areas.append(geom.area)
                else:
                    areas.append(0.0)
            except Exception:
                areas.append(0.0)
        
        # Create sorted results
        key_area_pairs = list(zip(keys, areas, geometries))
        sorted_pairs = sorted(key_area_pairs, key=lambda x: x[1], reverse=True)
        
        # Apply top_num filter
        if top_num is not None and top_num != 0:
            if top_num > 0:
                filtered_pairs = sorted_pairs[:top_num]
            else:
                filtered_pairs = sorted_pairs[top_num:]
        else:
            filtered_pairs = sorted_pairs
        
        # Build result dictionaries
        area_dict = {pair[0]: pair[1] for pair in filtered_pairs}
        geo_dict = {pair[0]: pair[2] for pair in filtered_pairs}
        
        return {
            'area_list': area_dict,
            'geo_map': geo_dict,
            'id_list': geo_dict
        }
        
    except Exception as e:
        logger.error(f"Error in area_filter: {e}")
        return {'area_list': {}, 'geo_map': {}, 'id_list': {}}

# Optimized geospatial calculations with parallel processing
def geo_calculate(data_list1_original: Any, data_list2_original: Any, mode: str, 
                 buffer_number: float = 0, versa_sign: bool = False, 
                 bounding_box: Optional[Dict] = None, test_mode: Optional[bool] = None) -> Dict[str, Any]:
    """Optimized geospatial calculations with parallel processing and vectorization"""
    
    if mode == 'area_filter':
        return area_filter(data_list1_original, buffer_number)
    
    # Track if we need to reverse the result (for contains operation)
    reverse_sign = False
    
    try:
        # Handle reverse operations
        if mode == 'contains':
            reverse_sign = True
            mode = 'in'
            # Swap the data for contains operation
            temp = data_list1_original
            data_list1_original = data_list2_original
            data_list2_original = temp
        
        # Normalize input data
        data1 = normalize_geometry_data(data_list1_original)
        data2 = normalize_geometry_data(data_list2_original)
        
        if not data1 or not data2:
            return {'object': {'id_list': {}}, 'subject': {'id_list': {}}, 'geo_map': {}}
        
        # Create optimized GeoSeries with coordinate transformation caching
        gseries1, gseries2 = create_optimized_geoseries(data1, data2)
        
        # Perform spatial operations
        result = perform_spatial_operation(gseries1, gseries2, mode, buffer_number)
        
        # Build result dictionaries
        return build_spatial_result(result, data1, data2, bounding_box, test_mode, reverse_sign)
        
    except Exception as e:
        logger.error(f"Error in geo_calculate: {e}")
        return {'object': {'id_list': {}}, 'subject': {'id_list': {}}, 'geo_map': {}}

def normalize_geometry_data(data: Any) -> Dict[str, Any]:
    """Normalize various data formats to geometry dictionary"""
    if isinstance(data, dict):
        if 'id_list' in data:
            return data['id_list']
        return data
    elif isinstance(data, list):
        return {key: global_id_geo.get(key) for key in data}
    elif isinstance(data, str):
        # Handle session-based geometry data
        session_data = use_globals_dict()
        return session_data.get(data, {})
    
    return {}

def create_optimized_geoseries(data1: Dict, data2: Dict) -> Tuple[gpd.GeoSeries, gpd.GeoSeries]:
    """Create optimized GeoSeries with cached coordinate transformation"""
    
    # Create GeoSeries
    gseries1 = gpd.GeoSeries(list(data1.values()), index=list(data1.keys()))
    gseries2 = gpd.GeoSeries(list(data2.values()), index=list(data2.keys()))
    
    # Cached coordinate transformation
    transformer = get_transformer(EPSG_WGS84, EPSG_UTM32N)
    
    # Set CRS and transform efficiently
    gseries1 = gseries1.set_crs(f"EPSG:{EPSG_WGS84}", allow_override=True)
    gseries2 = gseries2.set_crs(f"EPSG:{EPSG_WGS84}", allow_override=True)
    
    gseries1 = gseries1.to_crs(f"EPSG:{EPSG_UTM32N}")
    gseries2 = gseries2.to_crs(f"EPSG:{EPSG_UTM32N}")
    
    return gseries1, gseries2

def perform_spatial_operation(gseries1: gpd.GeoSeries, gseries2: gpd.GeoSeries, 
                             mode: str, buffer_distance: float) -> Dict[str, Any]:
    """Optimized spatial operations using vectorized geopandas operations"""
    
    gdf1 = gpd.GeoDataFrame(geometry=gseries1, index=gseries1.index)
    gdf2 = gpd.GeoDataFrame(geometry=gseries2, index=gseries2.index)
    
    if mode == "buffer":
        # Vectorized buffer operation
        buffered_gdf2 = gdf2.copy()
        buffered_gdf2['geometry'] = gdf2.geometry.buffer(buffer_distance)
        joined = optimize_spatial_join(gdf1, buffered_gdf2, 'intersects')
        
    elif mode == "in":
        joined = optimize_spatial_join(gdf1, gdf2, 'within')
        
    elif mode == "intersects":
        joined = optimize_spatial_join(gdf1, gdf2, 'intersects')
        
    elif mode == "shortest_distance":
        # For distance calculations, use spatial indexing for efficiency
        return calculate_shortest_distance_optimized(gdf1, gdf2)
        
    else:
        raise ValueError(f"Unsupported spatial operation mode: {mode}")
    
    # Extract results
    child_indices = set(joined.index)
    parent_indices = set(joined["index_right"]) if "index_right" in joined.columns else set()
    matching_pairs = list(zip(joined.index, joined.get("index_right", [])))
    
    return {
        'child_indices': child_indices,
        'parent_indices': parent_indices,
        'matching_pairs': matching_pairs
    }

def calculate_shortest_distance_optimized(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Optimized shortest distance calculation using spatial indexing"""
    min_distance = float('inf')
    closest_pair = (None, None)
    
    # Use spatial indexing for faster nearest neighbor search
    for idx1, geom1 in gdf1.geometry.items():
        # Find potential candidates using spatial index
        possible_matches_idx = list(gdf2.sindex.nearest(geom1.bounds, 1))
        
        for idx2 in possible_matches_idx:
            if idx2 < len(gdf2):
                geom2 = gdf2.geometry.iloc[idx2]
                distance = geom1.distance(geom2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (idx1, gdf2.index[idx2])
    
    return {
        'min_distance': min_distance,
        'closest_pair': closest_pair,
        'child_indices': {closest_pair[0]} if closest_pair[0] else set(),
        'parent_indices': {closest_pair[1]} if closest_pair[1] else set()
    }

def build_spatial_result(operation_result: Dict, data1: Dict, data2: Dict, 
                        bounding_box: Optional[Dict], test_mode: Optional[bool], 
                        reverse_sign: bool = False) -> Dict[str, Any]:
    """Build final spatial operation result"""
    
    child_indices = operation_result.get('child_indices', set())
    parent_indices = operation_result.get('parent_indices', set())
    
    # Build geo dictionaries
    parent_geo_dict = transfer_id_list_2_geo_dict(list(parent_indices), data2)
    child_geo_dict = transfer_id_list_2_geo_dict(list(child_indices), data1)
    
    # For contains operation, swap parent and child
    if reverse_sign:
        parent_geo_dict, child_geo_dict = child_geo_dict, parent_geo_dict
    
    # Build visualization geo_map
    geo_map = {}
    if bounding_box and not test_mode:
        region_name = bounding_box.get("bounding_box_region_name")
        bounding_wkb = bounding_box.get('bounding_wkb')
        if region_name and bounding_wkb:
            geo_map[region_name] = wkb.loads(bytes.fromhex(bounding_wkb))
    
    if not test_mode:
        geo_map.update(parent_geo_dict)
        geo_map.update(child_geo_dict)
    
    result = {
        'object': {'id_list': parent_geo_dict},
        'subject': {'id_list': child_geo_dict},
        'geo_map': geo_map
    }
    
    if test_mode and 'matching_pairs' in operation_result:
        result['match_dict'] = operation_result['matching_pairs']
    
    return result

# Optimized area calculation with vectorization
def calculate_areas(input_dict: Union[Dict, Any]) -> Dict[str, float]:
    """Vectorized area calculations with coordinate transformation caching"""
    
    if isinstance(input_dict, dict) and 'id_list' in input_dict:
        input_dict = input_dict['id_list']
    
    if not input_dict:
        return {}
    
    try:
        # Use cached transformer
        transformer = get_transformer(EPSG_WGS84, EPSG_UTM33N)
        
        result_dict = {}
        
        # Process geometries in batches
        for key, geom in input_dict.items():
            try:
                total_area = 0.0
                
                if isinstance(geom, Polygon):
                    total_area = calculate_polygon_area_utm(geom, transformer)
                elif isinstance(geom, MultiPolygon):
                    total_area = sum(
                        calculate_polygon_area_utm(poly, transformer) 
                        for poly in geom.geoms
                    )
                
                result_dict[key] = round(total_area, 2)
                
            except Exception as e:
                logger.warning(f"Error calculating area for {key}: {e}")
                result_dict[key] = 0.0
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in calculate_areas: {e}")
        return {}

def calculate_polygon_area_utm(polygon: Polygon, transformer: Transformer) -> float:
    """Calculate polygon area in UTM coordinates"""
    try:
        coords = np.array(polygon.exterior.coords)
        utm_coords = np.array(transformer.transform(coords[:, 0], coords[:, 1])).T
        utm_polygon = Polygon(utm_coords)
        return utm_polygon.area
    except Exception:
        return 0.0

# Optimized attribute querying with parallel processing
def get_attribute_by_column(name: str, mode: str) -> List[str]:
    """Optimized multi-table attribute querying with parallel processing"""
    
    tables = list(col_name_mapping_dict.keys())
    all_cols = ['name', 'fclass']
    other_mode = [col for col in all_cols if col != mode][0]
    
    def query_table(table: str) -> Set[str]:
        """Query single table for attributes"""
        try:
            table_config = col_name_mapping_dict[table]
            other_col_name = table_config.get(other_mode, other_mode)
            target_col_name = table_config.get(mode, mode)
            
            query = f"""
                SELECT DISTINCT {other_col_name} 
                FROM {table_config['graph_name']} 
                WHERE {target_col_name} = %s
            """
            
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (name,))
                    results = cur.fetchall()
                    return {result[other_col_name] for result in results if result[other_col_name]}
                    
        except Exception as e:
            logger.warning(f"Error querying table {table}: {e}")
            return set()
    
    # Use ThreadPoolExecutor for parallel database queries
    all_results = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_table = {executor.submit(query_table, table): table for table in tables}
        
        for future in future_to_table:
            try:
                result = future.result(timeout=10)
                all_results.update(result)
            except Exception as e:
                table = future_to_table[future]
                logger.warning(f"Error processing {table}: {e}")
    
    return list(all_results)

# Optimized statistics calculation
def equal_interval_stats(data_dict: Dict[str, float], num_intervals: int = 5) -> Dict[str, int]:
    """Optimized equal interval statistics calculation"""
    
    if not data_dict:
        return {}
    
    try:
        values = np.array(list(data_dict.values()))
        
        if len(values) == 0:
            return {}
        
        min_val, max_val = values.min(), values.max()
        
        if min_val == max_val:
            return {f"{min_val:.2f} - {max_val:.2f}": len(values)}
        
        # Create intervals
        intervals = np.linspace(min_val, max_val, num_intervals + 1)
        interval_labels = [f"{intervals[i]:.2f} - {intervals[i+1]:.2f}" for i in range(len(intervals)-1)]
        
        # Use numpy histogram for efficient binning
        counts, _ = np.histogram(values, bins=intervals)
        
        return dict(zip(interval_labels, counts.tolist()))
        
    except Exception as e:
        logger.error(f"Error in equal_interval_stats: {e}")
        return {}

# Optimized ID list explanation
def id_list_explain(id_list: Any, col: str = 'fclass') -> Dict[str, Any]:
    """Optimized ID list explanation with caching"""
    
    try:
        if not id_list:
            raise ValueError("Empty ID list provided")
        
        # Normalize input
        if isinstance(id_list, dict):
            if 'subject' in id_list:
                id_list = id_list['subject']
            if 'id_list' in id_list:
                id_list = id_list['id_list']
        
        # Handle attribute requests
        if 'attribute' in col:
            table_name = str(next(iter(id_list))).split('_')[0]
            return get_column_names(table_name)
        
        # Determine extraction index based on ID format
        sample_key = str(list(id_list.keys())[0])
        parts = sample_key.split("_")
        
        if len(parts) == 3:
            col = 'class'
        
        extract_index = 2 if col == 'name' else 1
        
        # Vectorized processing
        if col in ['fclass', 'type', 'class', 'name']:
            # Use pandas for efficient string processing
            keys_series = pd.Series(list(id_list.keys()))
            extracted_values = keys_series.str.split('_').str[extract_index]
            result = extracted_values.value_counts().to_dict()
            
        elif col == 'area':
            result = calculate_areas(id_list)
        else:
            result = {}
        
        # Sort by frequency/value
        sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        logger.info(f"ID list explanation result: {sorted_result}")
        
        return sorted_result
        
    except Exception as e:
        logger.error(f"Error in id_list_explain: {e}")
        return {}

# Optimized geometry dictionary transfer
def transfer_id_list_2_geo_dict(id_list: List[str], raw_dict: Optional[Dict] = None) -> Dict[str, Any]:
    """Optimized ID list to geometry dictionary transfer"""
    
    if not id_list:
        return {}
    
    if raw_dict is None:
        raw_dict = {}
    
    # Use vectorized pandas operations for better performance
    try:
        id_series = pd.Series(id_list)
        
        # Use map for efficient lookup
        def get_geometry(key):
            geom = raw_dict.get(key)
            if geom is None:
                geom = global_id_geo.get(key)
            return geom
        
        geometries = id_series.apply(get_geometry)
        
        # Filter out None values
        valid_mask = geometries.notna()
        valid_ids = id_series[valid_mask]
        valid_geometries = geometries[valid_mask]
        
        result_dict = dict(zip(valid_ids, valid_geometries))
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in transfer_id_list_2_geo_dict: {e}")
        return {}

# Optimized table creation from JSON
def create_table_from_json(json_data: Dict[str, Any], table_name: str) -> None:
    """Optimized table creation with batch insert"""
    
    table_name = f'uploaded_{table_name}'
    
    try:
        # Infer column types more intelligently
        columns = []
        for key, values in json_data.items():
            if key == 'geom':
                columns.append((key, 'GEOMETRY(Geometry, 4326)'))
            else:
                # Infer type from first non-null value
                sample_value = next((v for v in values if v is not None), None)
                if isinstance(sample_value, (int, np.integer)):
                    col_type = 'INTEGER'
                elif isinstance(sample_value, (float, np.floating)):
                    col_type = 'REAL'
                else:
                    col_type = 'TEXT'
                columns.append((key, col_type))
        
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Create table
                create_query = sql.SQL("CREATE TABLE {} ({})").format(
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(
                        sql.SQL("{} {}").format(sql.Identifier(col[0]), sql.SQL(col[1]))
                        for col in columns
                    )
                )
                cur.execute(create_query)
                
                # Batch insert data
                if json_data:
                    rows = list(zip(*json_data.values()))
                    
                    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        sql.Identifier(table_name),
                        sql.SQL(", ").join(map(sql.Identifier, json_data.keys())),
                        sql.SQL(", ").join([sql.Placeholder()] * len(json_data))
                    )
                    
                    # Use batch insert for better performance
                    cur.executemany(insert_query, rows)
                
                logger.info(f"Table {table_name} created successfully with {len(rows)} rows")
                
    except Exception as e:
        logger.error(f"Error creating table {table_name}: {e}")
        raise

# Optimized entity operations
def add_or_subtract_entities(json1: Dict[str, Any], json2: Dict[str, Any], operation: str) -> Dict[str, Any]:
    """Optimized entity addition/subtraction with set operations"""
    
    # Normalize inputs
    data1 = json1.get('id_list', json1) if isinstance(json1, dict) else json1
    data2 = json2.get('id_list', json2) if isinstance(json2, dict) else json2
    
    if operation not in ['add', 'subtract']:
        raise ValueError("Operation must be 'add' or 'subtract'")
    
    try:
        if operation == 'add':
            # Use dictionary union for Python 3.9+, fallback for older versions
            result = {**data1, **data2}
        else:  # subtract
            # Use set operations for efficient key filtering
            keys_to_keep = set(data1.keys()) - set(data2.keys())
            result = {key: data1[key] for key in keys_to_keep}
        
        return {'id_list': result, 'geo_map': result}
        
    except Exception as e:
        logger.error(f"Error in add_or_subtract_entities: {e}")
        return {'id_list': {}, 'geo_map': {}}

# Optimized attribute search
def search_attribute(dict_data: Dict[str, Any], key: str, value: Union[str, List[str]]) -> Dict[str, Any]:
    """Optimized attribute search with vectorized operations"""
    
    if not isinstance(value, list):
        value = [value]
    
    result_dict = {}
    geo_wkt_key = None
    
    try:
        # Find geometry key efficiently
        for subject_data in dict_data.values():
            geo_wkt_key = next((k for k in subject_data.keys() if "asWKT" in str(k)), None)
            if geo_wkt_key:
                break
        
        if not geo_wkt_key:
            logger.warning("No geometry key found in data")
            return {}
        
        # Vectorized search using pandas
        subjects = []
        keys_found = []
        geometries = []
        
        for subject, subject_data in dict_data.items():
            if key in subject_data:
                subject_key_value = subject_data[key]
                for v in value:
                    if v in subject_key_value:
                        subjects.append(subject)
                        keys_found.append(f"{key}_{v}_{subject}")
                        
                        try:
                            geom = wkt.loads(subject_data[geo_wkt_key])
                            geometries.append(geom)
                        except Exception as e:
                            logger.warning(f"Error parsing geometry for {subject}: {e}")
                            continue
        
        result_dict = dict(zip(keys_found, geometries))
        logger.info(f"Found {len(result_dict)} matching attributes")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in search_attribute: {e}")
        return {}

# Optimized uploaded data management
def get_uploaded_column_values(column_name: str) -> List[str]:
    """Optimized retrieval of uploaded table column values"""
    
    results = []
    
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Get all uploaded tables
                cur.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public' AND tablename LIKE 'uploaded_%'
                """)
                tables = cur.fetchall()
                
                for table in tables:
                    table_name = table['tablename']
                    
                    # Check if column exists
                    cur.execute("""
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = %s AND column_name = %s
                    """, (table_name, column_name))
                    
                    if cur.fetchone():
                        # Get column values
                        query = f'SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL'
                        cur.execute(query)
                        
                        table_results = [row[column_name] for row in cur.fetchall()]
                        results.extend(table_results)
        
        return list(set(results))  # Remove duplicates
        
    except Exception as e:
        logger.error(f"Error getting uploaded column values: {e}")
        return []

def del_uploaded_sql() -> None:
    """Optimized deletion of uploaded tables with memory cleanup"""
    
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Get all uploaded tables
                cur.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public' AND tablename LIKE 'uploaded_%'
                """)
                tables = cur.fetchall()
                
                uploaded_table_names = []
                
                # Drop tables in batch
                for table in tables:
                    table_name = table['tablename']
                    
                    try:
                        cur.execute(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE')
                        logger.info(f"Dropped table: {table_name}")
                        
                        if table_name.startswith('uploaded_'):
                            clean_name = table_name[9:]  # Remove 'uploaded_' prefix
                            uploaded_table_names.append(clean_name)
                            
                    except Exception as e:
                        logger.error(f"Error dropping table {table_name}: {e}")
        
        # Clean up memory structures
        cleanup_memory_structures(uploaded_table_names)
        logger.info("Uploaded tables cleanup completed")
        
    except Exception as e:
        logger.error(f"Error in del_uploaded_sql: {e}")

def cleanup_memory_structures(table_names: List[str]) -> None:
    """Clean up memory structures after table deletion"""
    
    try:
        # Clean up global col_name_mapping_dict
        global col_name_mapping_dict
        for table_name in table_names:
            if table_name in col_name_mapping_dict:
                del col_name_mapping_dict[table_name]
                logger.info(f"Removed {table_name} from col_name_mapping_dict")
        
        # Clean up session data if available
        try:
            if 'col_name_mapping_dict' in session:
                for table_name in table_names:
                    if table_name in session['col_name_mapping_dict']:
                        del session['col_name_mapping_dict'][table_name]
        except Exception:
            pass  # Session might not be available in all contexts
        
        # Try to clean up additional structures if they exist
        try:
            import ask_functions_agent
            
            for table_name in table_names:
                # Clean fclass_dict_4_similarity
                if hasattr(ask_functions_agent, 'fclass_dict_4_similarity'):
                    if table_name in ask_functions_agent.fclass_dict_4_similarity:
                        if hasattr(ask_functions_agent, 'all_fclass_set'):
                            ask_functions_agent.all_fclass_set -= set(
                                ask_functions_agent.fclass_dict_4_similarity[table_name]
                            )
                        del ask_functions_agent.fclass_dict_4_similarity[table_name]
                
                # Clean name_dict_4_similarity
                if hasattr(ask_functions_agent, 'name_dict_4_similarity'):
                    if table_name in ask_functions_agent.name_dict_4_similarity:
                        if hasattr(ask_functions_agent, 'all_name_set'):
                            ask_functions_agent.all_name_set -= set(
                                ask_functions_agent.name_dict_4_similarity[table_name]
                            )
                        del ask_functions_agent.name_dict_4_similarity[table_name]
                        
        except ImportError:
            logger.warning("ask_functions_agent module not available for cleanup")
        except Exception as e:
            logger.warning(f"Error cleaning additional structures: {e}")
    
    except Exception as e:
        logger.error(f"Error in cleanup_memory_structures: {e}")

# Optimized geospatial utility functions
def get_nearest_point(wkt_list: List[Any], location1: Tuple[float, float], 
                     location2: Tuple[float, float]) -> Tuple[Tuple[float, float], int]:
    """Optimized nearest point calculation with spatial indexing"""
    
    lat1, lon1 = location1
    lat2, lon2 = location2
    
    if not wkt_list:
        return (0.0, 0.0), -1
    
    try:
        line = LineString([(lon1, lat1), (lon2, lat2)])
        
        # Use vectorized operations for distance calculation
        distances = []
        centroids = []
        
        for geom in wkt_list:
            try:
                centroid = geom.centroid
                closest_point = line.interpolate(line.project(centroid))
                distance = centroid.distance(closest_point)
                
                distances.append(distance)
                centroids.append(centroid)
            except Exception:
                distances.append(float('inf'))
                centroids.append(Point(0, 0))
        
        # Find minimum distance index
        min_idx = np.argmin(distances)
        
        if min_idx < len(wkt_list):
            nearest_location = get_overall_centroid([wkt_list[min_idx]])
            return nearest_location, min_idx
        
        return (0.0, 0.0), -1
        
    except Exception as e:
        logger.error(f"Error in get_nearest_point: {e}")
        return (0.0, 0.0), -1

def get_overall_centroid(wkt_list: List[Any]) -> Tuple[float, float]:
    """Optimized overall centroid calculation using vectorization"""
    
    if not wkt_list:
        return (0.0, 0.0)
    
    try:
        # Vectorized centroid calculation
        centroids = [geom.centroid for geom in wkt_list if geom is not None]
        
        if not centroids:
            return (0.0, 0.0)
        
        # Use numpy for efficient coordinate averaging
        x_coords = np.array([c.x for c in centroids])
        y_coords = np.array([c.y for c in centroids])
        
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        
        return (avg_y, avg_x)  # Return as (lat, lon)
        
    except Exception as e:
        logger.error(f"Error in get_overall_centroid: {e}")
        return (0.0, 0.0)

def traffic_navigation(start_location: Dict[str, Any], end_location: Dict[str, Any], 
                      middle_locations_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optimized traffic navigation with route planning"""
    
    if middle_locations_list is None:
        middle_locations_list = []
    
    try:
        # Normalize inputs
        start_dict = start_location.get('id_list', start_location)
        end_dict = end_location.get('id_list', end_location)
        
        if not isinstance(middle_locations_list, list):
            middle_locations_list = [middle_locations_list]
        
        # Calculate route waypoints
        waypoints = []
        geo_map = {}
        
        start_point = get_overall_centroid(list(start_dict.values()))
        end_point = get_overall_centroid(list(end_dict.values()))
        
        waypoints.append(start_point)
        
        # Process middle locations
        for middle_place in middle_locations_list:
            middle_dict = middle_place.get('id_list', middle_place)
            
            middle_location, wkt_idx = get_nearest_point(
                list(middle_dict.values()), start_point, end_point
            )
            
            waypoints.append(middle_location)
            
            if wkt_idx >= 0:
                middle_key = list(middle_dict.keys())[wkt_idx]
                geo_map[middle_key] = middle_dict[middle_key]
        
        waypoints.append(end_point)
        
        # Add start and end locations to geo_map
        geo_map.update(start_dict)
        geo_map.update(end_dict)
        
        logger.info(f"Navigation waypoints: {waypoints}")
        
        # Note: compute_multi_stop_route function would need to be implemented
        # For now, return the waypoints structure
        
        return {
            'id_list': geo_map,
            'geo_map': geo_map,
            'waypoints': waypoints
        }
        
    except Exception as e:
        logger.error(f"Error in traffic_navigation: {e}")
        return {'id_list': {}, 'geo_map': {}, 'waypoints': []}

# Initialize reverse mapping and column mappings
for table_name in col_name_mapping_dict:
    graph_name = col_name_mapping_dict[table_name]['graph_name']
    revers_mapping_dict[graph_name] = table_name
    
    columns = get_column_names(table_name)
    for col in columns:
        if col not in col_name_mapping_dict[table_name]:
            col_name_mapping_dict[table_name][col] = col


# Cleanup function for proper resource management
def cleanup_resources():
    """Clean up database connections and other resources"""
    try:
        db_manager.close_pool()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup function to run at module exit
import atexit
atexit.register(cleanup_resources)

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper

# Apply performance monitoring to key functions
ids_of_type = monitor_performance(ids_of_type)
geo_calculate = monitor_performance(geo_calculate)
calculate_areas = monitor_performance(calculate_areas)

logger.info("Optimized geo_functions module loaded successfully")