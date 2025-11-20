"""
Extract nodes and edges from Stardog using SPARQL queries.
This module queries Stardog for RDF data and converts it to pandas DataFrames.
"""

import pandas as pd
import stardog


def get_stardog_connection(endpoint: str, username: str, password: str, database: str):
    """
    Create a connection to Stardog.
    
    Args:
        endpoint: Stardog server endpoint (e.g., 'http://localhost:5820')
        username: Stardog username
        password: Stardog password
        database: Database name
    
    Returns:
        stardog.Connection object
    """
    conn_details = {
        'endpoint': endpoint,
        'username': username,
        'password': password
    }
    return stardog.Connection(database, **conn_details)


def extract_nodes(conn: stardog.Connection) -> pd.DataFrame:
    """
    Extract all nodes (entities) from Stardog with their types and attributes.
    
    Returns:
        DataFrame with columns: uri, type, capacity, throughput
    """
    # SPARQL query to get all nodes with their types and attributes
    # Based on the Graph_Neural_Network ontology
    # Try querying from the data graph, or default graph
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    SELECT ?uri ?type ?capacity ?throughput
    WHERE {
        GRAPH <urn:data> {
            ?uri rdf:type ?type .
            FILTER (?type IN (gnn:OilField, gnn:Pipeline, gnn:Refinery, gnn:StorageTerminal, gnn:CustomerRegion))
            OPTIONAL { ?uri gnn:capacity_1 ?capacity . }
            OPTIONAL { ?uri gnn:throughput_1 ?throughput . }
        }
    }
    """
    
    # pystardog select() returns an iterable - need to handle it properly
    query_simple = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    SELECT ?uri ?type ?capacity ?throughput
    WHERE {
        ?uri rdf:type ?type .
        FILTER (?type IN (gnn:OilField, gnn:Pipeline, gnn:Refinery, gnn:StorageTerminal, gnn:CustomerRegion))
        OPTIONAL { ?uri gnn:capacity_1 ?capacity . }
        OPTIONAL { ?uri gnn:throughput_1 ?throughput . }
    }
    """
    
    # pystardog select() returns Union[bytes, Dict]
    # For JSON, it returns a dict with 'head' and 'results' keys
    # Use FROM clause to query the named graph <urn:data>
    query_with_graph = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    SELECT ?uri ?type ?capacity ?throughput
    FROM <urn:data>
    WHERE {
        ?uri rdf:type ?type .
        FILTER (?type IN (gnn:OilField, gnn:Pipeline, gnn:Refinery, gnn:StorageTerminal, gnn:CustomerRegion))
        OPTIONAL { ?uri gnn:capacity_1 ?capacity . }
        OPTIONAL { ?uri gnn:throughput_1 ?throughput . }
    }
    """
    results = conn.select(query_with_graph, content_type='application/sparql-results+json')
    
    rows = []
    if isinstance(results, dict):
        # JSON format: results['results']['bindings'] contains the actual data
        bindings = results.get('results', {}).get('bindings', [])
        for binding in bindings:
            uri = binding.get('uri', {}).get('value', '')
            type_uri = binding.get('type', {}).get('value', '')
            capacity_val = binding.get('capacity', {}).get('value')
            throughput_val = binding.get('throughput', {}).get('value')
            
            if uri:
                type_name = type_uri.split(':')[-1] if ':' in type_uri else type_uri.split('/')[-1]
                try:
                    capacity = float(capacity_val) if capacity_val is not None else 0.0
                except (ValueError, TypeError):
                    capacity = 0.0
                try:
                    throughput = float(throughput_val) if throughput_val is not None else 0.0
                except (ValueError, TypeError):
                    throughput = 0.0
                
                rows.append({
                    'uri': uri,
                    'type': type_name,
                    'capacity': capacity,
                    'throughput': throughput
                })
    elif isinstance(results, bytes):
        # If bytes, try to decode as JSON
        import json
        try:
            results_dict = json.loads(results.decode('utf-8'))
            bindings = results_dict.get('results', {}).get('bindings', [])
            for binding in bindings:
                uri = binding.get('uri', {}).get('value', '')
                type_uri = binding.get('type', {}).get('value', '')
                capacity_val = binding.get('capacity', {}).get('value')
                throughput_val = binding.get('throughput', {}).get('value')
                
                if uri:
                    type_name = type_uri.split(':')[-1] if ':' in type_uri else type_uri.split('/')[-1]
                    rows.append({
                        'uri': uri,
                        'type': type_name,
                        'capacity': float(capacity_val) if capacity_val is not None else 0.0,
                        'throughput': float(throughput_val) if throughput_val is not None else 0.0
                    })
        except Exception as e:
            print(f"Error parsing bytes results: {e}")
    
    return pd.DataFrame(rows)
    
    return pd.DataFrame(rows)


def extract_edges(conn: stardog.Connection) -> pd.DataFrame:
    """
    Extract all edges (relationships) from Stardog.
    
    Based on the Graph_Neural_Network ontology relationships:
    - isSuppliedBy: OilField → Pipeline
    - transportsTo: Pipeline → Refinery
    - storesAt: Refinery → StorageTerminal
    - storesFor: StorageTerminal → CustomerRegion
    
    Returns:
        DataFrame with columns: source_uri, target_uri, relation
    """
    # SPARQL query to get all relationships
    # Based on the Graph_Neural_Network ontology
    # Try querying from the data graph, or default graph
    query = """
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    SELECT ?source ?target ?relation
    WHERE {
        GRAPH <urn:data> {
            ?source ?relation ?target .
            FILTER (?relation IN (gnn:isSuppliedBy, gnn:transportsTo, gnn:storesAt, gnn:storesFor))
        }
    }
    """
    
    # Use FROM clause to query the named graph <urn:data>
    query_with_graph = """
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    SELECT ?source ?target ?relation
    FROM <urn:data>
    WHERE {
        ?source ?relation ?target .
        FILTER (?relation IN (gnn:isSuppliedBy, gnn:transportsTo, gnn:storesAt, gnn:storesFor))
    }
    """
    
    results = conn.select(query_with_graph, content_type='application/sparql-results+json')
    
    rows = []
    if isinstance(results, dict):
        bindings = results.get('results', {}).get('bindings', [])
        for binding in bindings:
            source_uri = binding.get('source', {}).get('value', '')
            target_uri = binding.get('target', {}).get('value', '')
            relation_uri = binding.get('relation', {}).get('value', '')
            
            if source_uri and target_uri:
                relation_name = relation_uri.split(':')[-1] if ':' in relation_uri else relation_uri.split('/')[-1]
                rows.append({
                    'source_uri': source_uri,
                    'target_uri': target_uri,
                    'relation': relation_name
                })
    elif isinstance(results, bytes):
        import json
        try:
            results_dict = json.loads(results.decode('utf-8'))
            bindings = results_dict.get('results', {}).get('bindings', [])
            for binding in bindings:
                source_uri = binding.get('source', {}).get('value', '')
                target_uri = binding.get('target', {}).get('value', '')
                relation_uri = binding.get('relation', {}).get('value', '')
                
                if source_uri and target_uri:
                    relation_name = relation_uri.split(':')[-1] if ':' in relation_uri else relation_uri.split('/')[-1]
                    rows.append({
                        'source_uri': source_uri,
                        'target_uri': target_uri,
                        'relation': relation_name
                    })
        except Exception as e:
            print(f"Error parsing bytes results: {e}")
    
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Stardog Cloud connection details
    ENDPOINT = 'https://presales.stardog.cloud:5820'
    USERNAME = 'aramcodemo2'
    PASSWORD = 'stardog'
    DATABASE = 'JF-gnn'
    
    print("Connecting to Stardog...")
    conn = get_stardog_connection(ENDPOINT, USERNAME, PASSWORD, DATABASE)
    
    print("Extracting nodes...")
    nodes_df = extract_nodes(conn)
    print(f"Found {len(nodes_df)} nodes")
    print(nodes_df.head())
    
    print("\nExtracting edges...")
    edges_df = extract_edges(conn)
    print(f"Found {len(edges_df)} edges")
    print(edges_df.head())
    
    # Save to CSV for inspection
    nodes_df.to_csv('../data/nodes_from_stardog.csv', index=False)
    edges_df.to_csv('../data/edges_from_stardog.csv', index=False)
    print("\nData saved to data/nodes_from_stardog.csv and data/edges_from_stardog.csv")
    
    conn.close()

