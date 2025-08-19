"""
Neo4j Graph Mapper Module

This module provides functions to query the Neo4j graph database
for text-to-entity mapping, replacing the original dictionary-based approach.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from functools import lru_cache
from neo4j import GraphDatabase, basic_auth
from config.database_config import NEO4J_CONFIG

logger = logging.getLogger(__name__)


class Neo4jMapper:
    """
    Handles mapping between text and database entities using Neo4j graph
    """
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
        """
        self.uri = uri or NEO4J_CONFIG['uri']
        self.user = user or NEO4J_CONFIG['user']
        self.password = password or NEO4J_CONFIG['password']
        self.driver = None
        self.connect()
    
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=basic_auth(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    @lru_cache(maxsize=1000)
    def get_fclass_values_for_table(self, table_name: str) -> Set[str]:
        """
        Get all fclass values for a specific table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Set of fclass values
        """
        with self.driver.session() as session:
            query = """
                MATCH (t:Table {name: $table_name})-[:TABLE_FCLASS]->(f:FClass)
                RETURN f.value as fclass
            """
            result = session.run(query, table_name=table_name)
            return {record['fclass'] for record in result if record['fclass']}
    
    @lru_cache(maxsize=1000)
    def get_name_values_for_table(self, table_name: str) -> Set[str]:
        """
        Get all name values for a specific table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Set of name values
        """
        with self.driver.session() as session:
            query = """
                MATCH (t:Table {name: $table_name})-[:TABLE_NAME]->(n:Name)
                RETURN n.value as name
            """
            result = session.run(query, table_name=table_name)
            return {record['name'] for record in result if record['name']}
    
    def get_all_fclass_values(self) -> Set[str]:
        """
        Get all unique fclass values across all tables
        
        Returns:
            Set of all fclass values
        """
        with self.driver.session() as session:
            query = """
                MATCH (f:FClass)
                RETURN DISTINCT f.value as fclass
            """
            result = session.run(query)
            return {record['fclass'] for record in result if record['fclass']}
    
    def get_all_name_values(self) -> Set[str]:
        """
        Get all unique name values across all tables
        
        Returns:
            Set of all name values
        """
        with self.driver.session() as session:
            query = """
                MATCH (n:Name)
                RETURN DISTINCT n.value as name
            """
            result = session.run(query)
            return {record['name'] for record in result if record['name']}
    
    def get_tables_for_fclass(self, fclass_value: str) -> List[str]:
        """
        Find tables that have a specific fclass value
        
        Args:
            fclass_value: The fclass value to search for
            
        Returns:
            List of table names
        """
        with self.driver.session() as session:
            query = """
                MATCH (f:FClass {value: $fclass_value})-[:FCLASS_TABLE]->(t:Table)
                RETURN t.name as table_name
            """
            result = session.run(query, fclass_value=fclass_value)
            return [record['table_name'] for record in result]
    
    def get_tables_for_name(self, name_value: str) -> List[str]:
        """
        Find tables that have a specific name value
        
        Args:
            name_value: The name value to search for
            
        Returns:
            List of table names
        """
        with self.driver.session() as session:
            query = """
                MATCH (n:Name {value: $name_value})-[:NAME_TABLE]->(t:Table)
                RETURN t.name as table_name
            """
            result = session.run(query, name_value=name_value)
            return [record['table_name'] for record in result]
    
    def fuzzy_search_fclass(self, search_term: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Perform fuzzy search on fclass values
        
        Args:
            search_term: The term to search for
            limit: Maximum number of results
            
        Returns:
            List of (fclass_value, score) tuples
        """
        with self.driver.session() as session:
            # Use fulltext index if available
            query = """
                CALL db.index.fulltext.queryNodes('fclass_fulltext', $search_term)
                YIELD node, score
                RETURN node.value as fclass, score
                ORDER BY score DESC
                LIMIT $limit
            """
            try:
                result = session.run(query, search_term=f"{search_term}~", limit=limit)
                return [(record['fclass'], record['score']) for record in result]
            except Exception:
                # Fallback to CONTAINS if fulltext index not available
                query = """
                    MATCH (f:FClass)
                    WHERE toLower(f.value) CONTAINS toLower($search_term)
                    RETURN f.value as fclass, 1.0 as score
                    LIMIT $limit
                """
                result = session.run(query, search_term=search_term, limit=limit)
                return [(record['fclass'], record['score']) for record in result]
    
    def fuzzy_search_name(self, search_term: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Perform fuzzy search on name values
        
        Args:
            search_term: The term to search for
            limit: Maximum number of results
            
        Returns:
            List of (name_value, score) tuples
        """
        with self.driver.session() as session:
            # Use fulltext index if available
            query = """
                CALL db.index.fulltext.queryNodes('name_fulltext', $search_term)
                YIELD node, score
                RETURN node.value as name, score
                ORDER BY score DESC
                LIMIT $limit
            """
            try:
                result = session.run(query, search_term=f"{search_term}~", limit=limit)
                return [(record['name'], record['score']) for record in result]
            except Exception:
                # Fallback to CONTAINS if fulltext index not available
                query = """
                    MATCH (n:Name)
                    WHERE toLower(n.value) CONTAINS toLower($search_term)
                    RETURN n.value as name, 1.0 as score
                    LIMIT $limit
                """
                result = session.run(query, search_term=search_term, limit=limit)
                return [(record['name'], record['score']) for record in result]
    
    
    def build_similarity_dictionaries(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Build dictionaries similar to fclass_dict_4_similarity and name_dict_4_similarity
        
        Returns:
            Tuple of (fclass_dict, name_dict)
        """
        fclass_dict = {}
        name_dict = {}
        
        with self.driver.session() as session:
            # Get all tables
            tables_result = session.run("MATCH (t:Table) RETURN t.name as table_name")
            tables = [record['table_name'] for record in tables_result]
            
            for table in tables:
                # Get fclass values for table
                fclass_dict[table] = self.get_fclass_values_for_table(table)
                
                # Get name values for table (skip soil)
                if table != 'soil':
                    name_dict[table] = self.get_name_values_for_table(table)
                else:
                    name_dict[table] = set()
        
        return fclass_dict, name_dict


# Global instance for easy access
_neo4j_mapper = None

def get_neo4j_mapper() -> Neo4jMapper:
    """
    Get global Neo4j mapper instance
    
    Returns:
        Neo4jMapper instance
    """
    global _neo4j_mapper
    if _neo4j_mapper is None:
        _neo4j_mapper = Neo4jMapper()
    return _neo4j_mapper

def close_neo4j_mapper():
    """Close global Neo4j mapper connection"""
    global _neo4j_mapper
    if _neo4j_mapper:
        _neo4j_mapper.close()
        _neo4j_mapper = None


# Compatibility functions to replace dictionary access
def get_fclass_dict_for_similarity() -> Dict[str, Set[str]]:
    """
    Get fclass dictionary for similarity matching
    Compatible replacement for fclass_dict_4_similarity
    """
    mapper = get_neo4j_mapper()
    fclass_dict, _ = mapper.build_similarity_dictionaries()
    return fclass_dict

def get_name_dict_for_similarity() -> Dict[str, Set[str]]:
    """
    Get name dictionary for similarity matching
    Compatible replacement for name_dict_4_similarity
    """
    mapper = get_neo4j_mapper()
    _, name_dict = mapper.build_similarity_dictionaries()
    return name_dict

def get_all_fclass_set() -> Set[str]:
    """
    Get all fclass values across all tables
    Compatible replacement for all_fclass_set
    """
    mapper = get_neo4j_mapper()
    return mapper.get_all_fclass_values()

def get_all_name_set() -> Set[str]:
    """
    Get all name values across all tables
    Compatible replacement for all_name_set
    """
    mapper = get_neo4j_mapper()
    return mapper.get_all_name_values()