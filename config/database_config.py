"""
Database Configuration Module

This module contains database connection parameters and table mapping configurations
for the geographic data processing system.
"""

# PostgreSQL connection parameters
POSTGRES_CONFIG = {
    "dbname": "osm_database",
    "user": "postgres",
    "host": "localhost",
    "password": "9417941"
}

POSTGRES_CONN_STRING = "dbname='osm_database' user='postgres' host='localhost' password='9417941'"

# Neo4j connection parameters
NEO4J_CONFIG = {
    "uri": "neo4j://127.0.0.1:7687",
    "user": "neo4j",
    "password": "9417941pqpqpq"
}

# EPSG coordinate system constants
EPSG_WGS84 = 4326
EPSG_UTM32N = 32632
EPSG_UTM33N = 32633

# Table name mappings for similar names
SIMILAR_TABLE_MAPPINGS = {
    'lands': "area", 
    'building': 'buildings', 
    'point': 'points',
    'soil': 'soil', 
    'areas': 'area', 
    'land': 'area'
}

# Column name mapping configuration
# This dictionary defines how logical column names map to actual database columns
# for each table in the system
COL_NAME_MAPPING_DICT = {
    "soil": {
        "osm_id": "objectid",      # OSM ID maps to objectid column
        "fclass": "leg_text",       # Classification maps to leg_text column
        "name": "leg_text",         # Name also maps to leg_text column
        "select_query": "SELECT leg_text,objectid,geom",
        "graph_name": "soilcomplete",  # Actual database table name
        "notice": "This Table Only has type Column."
    },
    "buildings": {
        "osm_id": "osm_id",
        "fclass": "type",           # Classification maps to type column
        "name": "name",
        "select_query": "SELECT 'buildings' AS source_table, type,name,osm_id,geom",
        "graph_name": "buildings"
    },
    "area": {
        "osm_id": "osm_id",
        "fclass": "fclass",         # Direct mapping to fclass column
        "name": "name",
        "select_query": "SELECT 'landuse' AS source_table, fclass,name,osm_id,geom",
        "graph_name": "landuse"     # Note: area maps to landuse table
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

# Build reverse mapping dictionary
REVERSE_MAPPING_DICT = {}
for table_name in COL_NAME_MAPPING_DICT:
    graph_name = COL_NAME_MAPPING_DICT[table_name]['graph_name']
    REVERSE_MAPPING_DICT[graph_name] = table_name

def get_table_config(table_name: str) -> dict:
    """
    Get configuration for a specific table
    
    Args:
        table_name: Logical table name
        
    Returns:
        Table configuration dictionary
    """
    return COL_NAME_MAPPING_DICT.get(table_name, {})

def get_actual_column_name(table_name: str, logical_column: str) -> str:
    """
    Get the actual database column name for a logical column
    
    Args:
        table_name: Logical table name
        logical_column: Logical column name (e.g., 'fclass', 'name')
        
    Returns:
        Actual database column name
    """
    table_config = COL_NAME_MAPPING_DICT.get(table_name, {})
    return table_config.get(logical_column, logical_column)

def get_actual_table_name(logical_table: str) -> str:
    """
    Get the actual database table name for a logical table
    
    Args:
        logical_table: Logical table name
        
    Returns:
        Actual database table name
    """
    table_config = COL_NAME_MAPPING_DICT.get(logical_table, {})
    return table_config.get('graph_name', logical_table)