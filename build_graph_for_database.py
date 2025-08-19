"""
Build Graph for Database Script

This script builds a Neo4j graph database from PostgreSQL data based on the 
column mapping configuration. The graph replaces the fclass_dict_4_similarity
and name_dict_4_similarity dictionaries for text-to-entity mapping.

The graph structure:
- Nodes: Database, Table, FClass, Name
- Relationships (bidirectional): 
  - DATABASE_TABLE / TABLE_DATABASE
  - TABLE_FCLASS / FCLASS_TABLE  
  - TABLE_NAME / NAME_TABLE
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
                session.run("CREATE INDEX database_name_idx IF NOT EXISTS FOR (d:Database) ON (d.name)")
                session.run("CREATE INDEX table_name_idx IF NOT EXISTS FOR (t:Table) ON (t.name)")
                session.run("CREATE INDEX fclass_value_idx IF NOT EXISTS FOR (f:FClass) ON (f.value)")
                session.run("CREATE INDEX name_value_idx IF NOT EXISTS FOR (n:Name) ON (n.value)")
                logger.info("Created indexes")
                
            except Exception as e:
                logger.error(f"Error clearing database: {e}")
                raise
    
    def create_database_node(self, database_name: str = "postgres"):
        """Create the database node"""
        with self.driver.session() as session:
            query = """
                MERGE (d:Database {name: $database_name})
                RETURN d
            """
            session.run(query, database_name=database_name)
            logger.info(f"Created database node: {database_name}")
    
    def create_table_node_with_relationship(self, table_name: str, database_name: str = "postgres"):
        """Create table node and link to database with bidirectional relationships"""
        with self.driver.session() as session:
            query = """
                MATCH (d:Database {name: $database_name})
                MERGE (t:Table {name: $table_name})
                MERGE (d)-[:DATABASE_TABLE]->(t)
                MERGE (t)-[:TABLE_DATABASE]->(d)
                RETURN t
            """
            session.run(query, table_name=table_name, database_name=database_name)
            logger.debug(f"Created table node with relationships: {table_name}")
    
    def create_fclass_nodes_batch(self, table_name: str, fclass_values: Set[str]):
        """Create FClass nodes with bidirectional relationships to table"""
        if not fclass_values:
            return
        
        with self.driver.session() as session:
            # Convert set to list for batch processing
            fclass_list = list(fclass_values)
            
            # Create FClass nodes and bidirectional relationships in batch
            query = """
                UNWIND $fclass_list AS fclass_value
                MERGE (f:FClass {value: fclass_value})
                WITH f, $table_name AS table_name
                MATCH (t:Table {name: table_name})
                MERGE (t)-[:TABLE_FCLASS]->(f)
                MERGE (f)-[:FCLASS_TABLE]->(t)
                RETURN count(DISTINCT f) as created_count
            """
            
            result = session.run(query, table_name=table_name, fclass_list=fclass_list)
            count = result.single()['created_count']
            logger.info(f"Created/linked {count} FClass nodes for table {table_name}")
    
    def create_name_nodes_batch(self, table_name: str, name_values: Set[str]):
        """Create Name nodes with bidirectional relationships to table"""
        if not name_values:
            return
        
        with self.driver.session() as session:
            # Convert set to list for batch processing
            name_list = list(name_values)
            
            # Create Name nodes and bidirectional relationships in batch
            query = """
                UNWIND $name_list AS name_value
                MERGE (n:Name {value: name_value})
                WITH n, $table_name AS table_name
                MATCH (t:Table {name: table_name})
                MERGE (t)-[:TABLE_NAME]->(n)
                MERGE (n)-[:NAME_TABLE]->(t)
                RETURN count(DISTINCT n) as created_count
            """
            
            result = session.run(query, table_name=table_name, name_list=name_list)
            count = result.single()['created_count']
            logger.info(f"Created/linked {count} Name nodes for table {table_name}")
    
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the graph"""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            for label in ['Database', 'Table', 'FClass', 'Name']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                stats[f'{label} nodes'] = result.single()['count']
            
            # Count relationships by type
            relationship_types = [
                'DATABASE_TABLE', 'TABLE_DATABASE',
                'TABLE_FCLASS', 'FCLASS_TABLE',
                'TABLE_NAME', 'NAME_TABLE'
            ]
            
            for rel_type in relationship_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                stats[f'{rel_type} relationships'] = result.single()['count']
            
            # Total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['Total relationships'] = result.single()['count']
            
            return stats


def build_graph_from_database():
    """
    Main function to build Neo4j graph from PostgreSQL database
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
        
        # Create database node
        logger.info("Creating database node...")
        neo4j.create_database_node("postgres")
        
        # Process each table
        logger.info("Building graph from database tables...")
        
        total_fclass_count = 0
        total_name_count = 0
        
        for logical_table, table_config in COL_NAME_MAPPING_DICT.items():
            logger.info(f"\nProcessing table: {logical_table}")
            
            actual_table = table_config.get('graph_name', logical_table)
            
            # Create table node with database relationship
            neo4j.create_table_node_with_relationship(logical_table, "postgres")
            
            # Fetch and create FClass nodes
            fclass_values = postgres.fetch_distinct_values(
                logical_table, 'fclass', actual_table
            )
            if fclass_values:
                neo4j.create_fclass_nodes_batch(logical_table, fclass_values)
                total_fclass_count += len(fclass_values)
            
            # Fetch and create Name nodes (skip for soil table)
            if logical_table != 'soil':
                name_values = postgres.fetch_distinct_values(
                    logical_table, 'name', actual_table
                )
                if name_values:
                    neo4j.create_name_nodes_batch(logical_table, name_values)
                    total_name_count += len(name_values)
        
        # Create text mapping indexes
        neo4j.create_text_mapping_indexes()
        
        # Display statistics
        stats = neo4j.get_statistics()
        logger.info("\n" + "="*60)
        logger.info("Graph Building Complete!")
        logger.info("="*60)
        logger.info("\nGraph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:,}")
        
        logger.info(f"\nSummary:")
        logger.info(f"  Total unique FClass values: {total_fclass_count:,}")
        logger.info(f"  Total unique Name values: {total_name_count:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up connections
        if postgres:
            postgres.close()
        if neo4j:
            neo4j.close()


def query_graph_examples():
    """Example queries to demonstrate graph usage"""
    neo4j = Neo4jGraphBuilder(
        NEO4J_CONFIG['uri'],
        NEO4J_CONFIG['user'],
        NEO4J_CONFIG['password']
    )
    
    try:
        with neo4j.driver.session() as session:
            print("\n" + "="*60)
            print("Example Graph Queries")
            print("="*60)
            
            # Example 1: Database to Tables
            print("\n1. Database and its tables:")
            result = session.run("""
                MATCH (d:Database)-[:DATABASE_TABLE]->(t:Table)
                RETURN d.name as database, collect(t.name) as tables
            """)
            for record in result:
                print(f"  {record['database']}: {', '.join(record['tables'])}")
            
            # Example 2: Table to FClass values
            print("\n2. FClass values for 'buildings' table:")
            result = session.run("""
                MATCH (t:Table {name: 'buildings'})-[:TABLE_FCLASS]->(f:FClass)
                RETURN f.value as fclass
                ORDER BY fclass
                LIMIT 10
            """)
            for record in result:
                print(f"  - {record['fclass']}")
            
            # Example 3: Reverse - FClass to Tables
            print("\n3. Tables that have 'residential' fclass:")
            result = session.run("""
                MATCH (f:FClass {value: 'residential'})-[:FCLASS_TABLE]->(t:Table)
                RETURN t.name as table_name
            """)
            for record in result:
                print(f"  - {record['table_name']}")
            
            # Example 4: Bidirectional traversal
            print("\n4. Path from Database to FClass 'park':")
            result = session.run("""
                MATCH path = (d:Database {name: 'postgres'})-[:DATABASE_TABLE]->(t:Table)-[:TABLE_FCLASS]->(f:FClass {value: 'park'})
                RETURN d.name as db, t.name as table, f.value as fclass
                LIMIT 5
            """)
            for record in result:
                print(f"  {record['db']} -> {record['table']} -> {record['fclass']}")
            
            # Example 5: Statistics
            print("\n5. Table statistics:")
            result = session.run("""
                MATCH (t:Table)
                OPTIONAL MATCH (t)-[:TABLE_FCLASS]->(f:FClass)
                OPTIONAL MATCH (t)-[:TABLE_NAME]->(n:Name)
                RETURN t.name as table, 
                       count(DISTINCT f) as fclass_count,
                       count(DISTINCT n) as name_count
                ORDER BY table
            """)
            for record in result:
                print(f"  {record['table']}: {record['fclass_count']} fclass, {record['name_count']} names")
    
    finally:
        neo4j.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build Neo4j graph from PostgreSQL database with bidirectional relationships"
    )
    parser.add_argument(
        "--query-examples",
        action="store_true",
        help="Run example queries after building"
    )
    
    args = parser.parse_args()
    
    # Build the graph
    success = build_graph_from_database()
    
    if success and args.query_examples:
        query_graph_examples()
    
    sys.exit(0 if success else 1)