"""
Build Graph for Database Script

This script builds a Neo4j graph database from PostgreSQL data based on the 
column mapping configuration. The graph replaces the fclass_dict_4_similarity
and name_dict_4_similarity dictionaries for text-to-entity mapping.

The graph structure:
- Nodes: Table, FClass, Name, Entity
- Relationships: HAS_FCLASS, HAS_NAME, BELONGS_TO
"""

import logging
import sys
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase, basic_auth
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append('.')

from config.database_config import (
    POSTGRES_CONFIG,
    NEO4J_CONFIG,
    COL_NAME_MAPPING_DICT,
    get_actual_column_name,
    get_actual_table_name
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgreSQLConnector:
    """Handle PostgreSQL database connections and queries"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL connection closed")
    
    def fetch_distinct_values(self, table_name: str, column_name: str, 
                            actual_table_name: str) -> Set[str]:
        """
        Fetch distinct values from a column
        
        Args:
            table_name: Logical table name
            column_name: Logical column name (fclass or name)
            actual_table_name: Actual database table name
            
        Returns:
            Set of distinct values
        """
        try:
            actual_column = get_actual_column_name(table_name, column_name)
            
            # Skip if column doesn't exist for this table
            if not actual_column:
                return set()
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if column exists
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    AND column_name = %s
                    AND table_schema = 'public'
                """, (actual_table_name, actual_column))
                
                if not cur.fetchone():
                    logger.warning(f"Column {actual_column} not found in table {actual_table_name}")
                    return set()
                
                # Fetch distinct values
                query = f"""
                    SELECT DISTINCT {actual_column} as value
                    FROM {actual_table_name}
                    WHERE {actual_column} IS NOT NULL
                    AND {actual_column} != ''
                    ORDER BY value
                """
                
                cur.execute(query)
                results = cur.fetchall()
                
                values = {row['value'] for row in results if row['value']}
                logger.info(f"Fetched {len(values)} distinct {column_name} values from {table_name}")
                
                return values
                
        except psycopg2.Error as e:
            logger.error(f"Error fetching {column_name} from {table_name}: {e}")
            return set()
    
    def fetch_entities_with_attributes(self, table_name: str, 
                                      actual_table_name: str,
                                      limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch entities with their fclass and name attributes
        
        Args:
            table_name: Logical table name
            actual_table_name: Actual database table name
            limit: Optional limit on number of entities
            
        Returns:
            List of entity dictionaries
        """
        try:
            table_config = COL_NAME_MAPPING_DICT.get(table_name, {})
            osm_id_col = table_config.get('osm_id', 'osm_id')
            fclass_col = table_config.get('fclass', 'fclass')
            name_col = table_config.get('name', 'name')
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query based on available columns
                select_parts = [f"{osm_id_col} as osm_id"]
                
                # Check and add fclass column
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = %s AND column_name = %s
                """, (actual_table_name, fclass_col))
                if cur.fetchone():
                    select_parts.append(f"{fclass_col} as fclass")
                
                # Check and add name column
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = %s AND column_name = %s
                """, (actual_table_name, name_col))
                if cur.fetchone():
                    select_parts.append(f"{name_col} as name")
                
                query = f"""
                    SELECT {', '.join(select_parts)}
                    FROM {actual_table_name}
                    WHERE {osm_id_col} IS NOT NULL
                    {f'LIMIT {limit}' if limit else ''}
                """
                
                cur.execute(query)
                results = cur.fetchall()
                
                logger.info(f"Fetched {len(results)} entities from {table_name}")
                return results
                
        except psycopg2.Error as e:
            logger.error(f"Error fetching entities from {table_name}: {e}")
            return []


class Neo4jGraphBuilder:
    """Build and manage Neo4j graph database"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        logger.info("Connected to Neo4j database")
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        with self.driver.session() as session:
            try:
                # Delete all relationships and nodes
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared Neo4j database")
                
                # Create indexes for performance
                session.run("CREATE INDEX table_name_idx IF NOT EXISTS FOR (t:Table) ON (t.name)")
                session.run("CREATE INDEX fclass_value_idx IF NOT EXISTS FOR (f:FClass) ON (f.value)")
                session.run("CREATE INDEX name_value_idx IF NOT EXISTS FOR (n:Name) ON (n.value)")
                session.run("CREATE INDEX entity_id_idx IF NOT EXISTS FOR (e:Entity) ON (e.osm_id)")
                logger.info("Created indexes")
                
            except Exception as e:
                logger.error(f"Error clearing database: {e}")
                raise
    
    def create_table_node(self, table_name: str):
        """Create a table node"""
        with self.driver.session() as session:
            query = """
                MERGE (t:Table {name: $table_name})
                RETURN t
            """
            session.run(query, table_name=table_name)
            logger.debug(f"Created table node: {table_name}")
    
    def create_fclass_nodes_batch(self, table_name: str, fclass_values: Set[str]):
        """Create FClass nodes and link to table in batch"""
        if not fclass_values:
            return
        
        with self.driver.session() as session:
            # Convert set to list for batch processing
            fclass_list = list(fclass_values)
            
            # Create FClass nodes and relationships in batch
            query = """
                UNWIND $fclass_list AS fclass_value
                MERGE (f:FClass {value: fclass_value})
                WITH f, $table_name AS table_name
                MATCH (t:Table {name: table_name})
                MERGE (t)-[:HAS_FCLASS]->(f)
                RETURN count(f) as created_count
            """
            
            result = session.run(query, table_name=table_name, fclass_list=fclass_list)
            count = result.single()['created_count']
            logger.info(f"Created/linked {count} FClass nodes for table {table_name}")
    
    def create_name_nodes_batch(self, table_name: str, name_values: Set[str]):
        """Create Name nodes and link to table in batch"""
        if not name_values:
            return
        
        with self.driver.session() as session:
            # Convert set to list for batch processing
            name_list = list(name_values)
            
            # Create Name nodes and relationships in batch
            query = """
                UNWIND $name_list AS name_value
                MERGE (n:Name {value: name_value})
                WITH n, $table_name AS table_name
                MATCH (t:Table {name: table_name})
                MERGE (t)-[:HAS_NAME]->(n)
                RETURN count(n) as created_count
            """
            
            result = session.run(query, table_name=table_name, name_list=name_list)
            count = result.single()['created_count']
            logger.info(f"Created/linked {count} Name nodes for table {table_name}")
    
    def create_entities_batch(self, table_name: str, entities: List[Dict], batch_size: int = 1000):
        """Create Entity nodes with relationships in batch"""
        if not entities:
            return
        
        with self.driver.session() as session:
            # Process in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                
                # Prepare entity data
                entity_data = []
                for entity in batch:
                    entity_dict = {
                        'osm_id': str(entity.get('osm_id', '')),
                        'table_name': table_name,
                        'fclass': entity.get('fclass'),
                        'name': entity.get('name')
                    }
                    entity_data.append(entity_dict)
                
                # Create entities and relationships
                query = """
                    UNWIND $entities AS entity_data
                    MERGE (e:Entity {
                        osm_id: entity_data.osm_id,
                        table_name: entity_data.table_name
                    })
                    WITH e, entity_data
                    MATCH (t:Table {name: entity_data.table_name})
                    MERGE (e)-[:BELONGS_TO]->(t)
                    WITH e, entity_data
                    FOREACH (fclass IN CASE WHEN entity_data.fclass IS NOT NULL 
                                       THEN [entity_data.fclass] ELSE [] END |
                        MERGE (f:FClass {value: fclass})
                        MERGE (e)-[:HAS_FCLASS]->(f)
                    )
                    WITH e, entity_data
                    FOREACH (name IN CASE WHEN entity_data.name IS NOT NULL 
                                     THEN [entity_data.name] ELSE [] END |
                        MERGE (n:Name {value: name})
                        MERGE (e)-[:HAS_NAME]->(n)
                    )
                    RETURN count(e) as created_count
                """
                
                result = session.run(query, entities=entity_data)
                count = result.single()['created_count']
                logger.debug(f"Created {count} entities in batch {i//batch_size + 1}")
    
    def create_text_mapping_indexes(self):
        """Create additional indexes for text mapping"""
        with self.driver.session() as session:
            try:
                # Create fulltext indexes for fuzzy matching
                session.run("""
                    CREATE FULLTEXT INDEX fclass_fulltext IF NOT EXISTS 
                    FOR (f:FClass) ON EACH [f.value]
                """)
                session.run("""
                    CREATE FULLTEXT INDEX name_fulltext IF NOT EXISTS 
                    FOR (n:Name) ON EACH [n.value]
                """)
                logger.info("Created fulltext indexes for text mapping")
            except Exception as e:
                logger.warning(f"Could not create fulltext indexes: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the graph"""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            for label in ['Table', 'FClass', 'Name', 'Entity']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                stats[label] = result.single()['count']
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['Relationships'] = result.single()['count']
            
            return stats


def build_graph_from_database(include_entities: bool = False, entity_limit: Optional[int] = None):
    """
    Main function to build Neo4j graph from PostgreSQL database
    
    Args:
        include_entities: If True, also create Entity nodes (can be memory intensive)
        entity_limit: Optional limit on entities per table (for testing)
    """
    postgres = None
    neo4j = None
    
    try:
        # Initialize connections
        logger.info("Initializing database connections...")
        postgres = PostgreSQLConnector(POSTGRES_CONFIG)
        neo4j = Neo4jGraphBuilder(
            NEO4J_CONFIG['uri'],
            NEO4J_CONFIG['user'],
            NEO4J_CONFIG['password']
        )
        
        # Clear Neo4j database
        logger.info("Clearing Neo4j database...")
        neo4j.clear_database()
        
        # Process each table
        logger.info("Building graph from database tables...")
        
        for logical_table, table_config in COL_NAME_MAPPING_DICT.items():
            logger.info(f"\nProcessing table: {logical_table}")
            
            actual_table = table_config.get('graph_name', logical_table)
            
            # Create table node
            neo4j.create_table_node(logical_table)
            
            # Fetch and create FClass nodes
            fclass_values = postgres.fetch_distinct_values(
                logical_table, 'fclass', actual_table
            )
            if fclass_values:
                neo4j.create_fclass_nodes_batch(logical_table, fclass_values)
            
            # Fetch and create Name nodes (skip for soil table)
            if logical_table != 'soil':
                name_values = postgres.fetch_distinct_values(
                    logical_table, 'name', actual_table
                )
                if name_values:
                    neo4j.create_name_nodes_batch(logical_table, name_values)
            
            # Optionally create entity nodes
            if include_entities:
                logger.info(f"Creating entity nodes for {logical_table}...")
                entities = postgres.fetch_entities_with_attributes(
                    logical_table, actual_table, entity_limit
                )
                if entities:
                    neo4j.create_entities_batch(logical_table, entities)
        
        # Create text mapping indexes
        neo4j.create_text_mapping_indexes()
        
        # Display statistics
        stats = neo4j.get_statistics()
        logger.info("\n=== Graph Building Complete ===")
        logger.info("Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        return False
        
    finally:
        # Clean up connections
        if postgres:
            postgres.close()
        if neo4j:
            neo4j.close()


def query_graph_example():
    """Example queries to demonstrate graph usage"""
    neo4j = Neo4jGraphBuilder(
        NEO4J_CONFIG['uri'],
        NEO4J_CONFIG['user'],
        NEO4J_CONFIG['password']
    )
    
    try:
        with neo4j.driver.session() as session:
            # Example 1: Find all fclass values for a table
            result = session.run("""
                MATCH (t:Table {name: 'buildings'})-[:HAS_FCLASS]->(f:FClass)
                RETURN f.value as fclass
                ORDER BY fclass
                LIMIT 10
            """)
            
            print("\nExample: FClass values for 'buildings' table:")
            for record in result:
                print(f"  - {record['fclass']}")
            
            # Example 2: Find tables that have a specific fclass value
            result = session.run("""
                MATCH (t:Table)-[:HAS_FCLASS]->(f:FClass {value: 'residential'})
                RETURN t.name as table_name
            """)
            
            print("\nExample: Tables with 'residential' fclass:")
            for record in result:
                print(f"  - {record['table_name']}")
            
            # Example 3: Fuzzy text search
            result = session.run("""
                CALL db.index.fulltext.queryNodes('fclass_fulltext', 'park~')
                YIELD node, score
                RETURN node.value as fclass, score
                ORDER BY score DESC
                LIMIT 5
            """)
            
            print("\nExample: Fuzzy search for 'park' in fclass:")
            for record in result:
                print(f"  - {record['fclass']} (score: {record['score']:.2f})")
    
    finally:
        neo4j.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build Neo4j graph from PostgreSQL database"
    )
    parser.add_argument(
        "--include-entities",
        action="store_true",
        help="Include entity nodes (warning: can be memory intensive)"
    )
    parser.add_argument(
        "--entity-limit",
        type=int,
        help="Limit number of entities per table (for testing)"
    )
    parser.add_argument(
        "--query-examples",
        action="store_true",
        help="Run example queries after building"
    )
    
    args = parser.parse_args()
    
    # Build the graph
    success = build_graph_from_database(
        include_entities=args.include_entities,
        entity_limit=args.entity_limit
    )
    
    if success and args.query_examples:
        query_graph_example()
    
    sys.exit(0 if success else 1)