"""
Evaluate predicted links with cost-benefit analysis.
This module analyzes predicted relationships to determine business value and implementation feasibility.
"""

import pandas as pd
import stardog
from extract_from_stardog import get_stardog_connection
from build_pyg_graph import load_from_csv
import networkx as nx
from typing import List, Dict, Tuple


def build_networkx_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a NetworkX graph from the nodes and edges DataFrames.
    Used for path analysis and network efficiency calculations.
    
    Args:
        nodes_df: DataFrame with node information
        edges_df: DataFrame with edge information
    
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        G.add_node(row['uri'], 
                   type=row['type'],
                   capacity=row.get('capacity', 0),
                   throughput=row.get('throughput', 0))
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['source_uri'], row['target_uri'], 
                   relation=row.get('relation', ''))
    
    return G


def calculate_path_efficiency(G: nx.DiGraph, source_uri: str, target_uri: str) -> Dict:
    """
    Calculate path efficiency metrics for a potential new link.
    
    Args:
        G: NetworkX graph
        source_uri: Source node URI
        target_uri: Target node URI
    
    Returns:
        Dictionary with path efficiency metrics
    """
    metrics = {
        'direct_path_exists': False,
        'shortest_path_length': None,
        'path_efficiency_gain': 0,
        'alternative_paths': []
    }
    
    # Check if direct path already exists
    if G.has_edge(source_uri, target_uri):
        metrics['direct_path_exists'] = True
        metrics['shortest_path_length'] = 1
        return metrics
    
    # Find shortest path without the new link
    try:
        shortest_path = nx.shortest_path(G, source_uri, target_uri)
        current_path_length = len(shortest_path) - 1  # Number of edges
        metrics['shortest_path_length'] = current_path_length
        
        # If we add this direct link, path length becomes 1
        if current_path_length > 1:
            metrics['path_efficiency_gain'] = current_path_length - 1
            metrics['alternative_paths'] = [shortest_path]
    except nx.NetworkXNoPath:
        # No path exists currently - this would be a new connection
        metrics['shortest_path_length'] = None
        metrics['path_efficiency_gain'] = 10  # High value for new connectivity
    
    return metrics


def estimate_revenue_potential(source_node: Dict, target_node: Dict, path_metrics: Dict) -> float:
    """
    Estimate revenue potential for a predicted link.
    
    Revenue factors:
    - Available capacity (more capacity = more revenue potential)
    - Market size (customer regions have higher revenue potential)
    - Path efficiency (faster delivery = better service = more revenue)
    - New market access (serving new region = new revenue stream)
    
    Args:
        source_node: Source node attributes
        target_node: Target node attributes
        path_metrics: Path efficiency metrics
    
    Returns:
        Estimated revenue potential score (0-100)
    """
    revenue_score = 0.0
    
    # Factor 1: Available capacity (40% weight)
    source_capacity = source_node.get('capacity', 0)
    source_throughput = source_node.get('throughput', 0)
    available_capacity = max(0, source_capacity - source_throughput)
    capacity_utilization = source_throughput / source_capacity if source_capacity > 0 else 0
    
    # More available capacity = higher revenue potential
    if source_capacity > 0:
        capacity_score = (available_capacity / source_capacity) * 100
        revenue_score += capacity_score * 0.4
    
    # Factor 2: Target market type (30% weight)
    target_type = target_node.get('type', '')
    if target_type == 'CustomerRegion':
        # Customer regions represent revenue opportunities
        market_score = 80  # High value for customer connections
    elif target_type in ['StorageTerminal', 'Refinery']:
        # Infrastructure connections enable revenue
        market_score = 50
    else:
        market_score = 30
    
    revenue_score += market_score * 0.3
    
    # Factor 3: Path efficiency gain (20% weight)
    path_gain = path_metrics.get('path_efficiency_gain', 0)
    if path_gain > 0:
        # More efficiency gain = better service = more revenue
        efficiency_score = min(100, path_gain * 20)  # Cap at 100
        revenue_score += efficiency_score * 0.2
    elif path_metrics.get('shortest_path_length') is None:
        # New connectivity = new market = high revenue potential
        revenue_score += 80 * 0.2
    
    # Factor 4: Capacity utilization (10% weight)
    # Moderate utilization (50-80%) is ideal - not too full, not too empty
    if 0.5 <= capacity_utilization <= 0.8:
        utilization_score = 100
    elif capacity_utilization < 0.5:
        utilization_score = 50  # Underutilized
    else:
        utilization_score = 30  # Overutilized
    
    revenue_score += utilization_score * 0.1
    
    return min(100, revenue_score)


def estimate_implementation_cost(source_node: Dict, target_node: Dict, path_metrics: Dict) -> float:
    """
    Estimate implementation cost for a predicted link.
    
    Cost factors:
    - Throughput requirements (higher throughput = higher cost)
    - Path complexity (longer paths = higher cost)
    - Infrastructure requirements
    
    Args:
        source_node: Source node attributes
        target_node: Target node attributes
        path_metrics: Path efficiency metrics
    
    Returns:
        Estimated cost score (0-100, lower is better/costlier)
    """
    cost_score = 100.0  # Start with low cost assumption
    
    # Factor 1: Throughput requirements (50% weight)
    source_throughput = source_node.get('throughput', 0)
    source_capacity = source_node.get('capacity', 0)
    
    # Higher throughput = higher operational costs
    if source_capacity > 0:
        throughput_ratio = source_throughput / source_capacity
        # Higher utilization = higher costs
        cost_penalty = throughput_ratio * 50
        cost_score -= cost_penalty * 0.5
    
    # Factor 2: Path complexity (30% weight)
    path_length = path_metrics.get('shortest_path_length')
    if path_length and path_length > 3:
        # Longer paths = more complex = higher cost
        cost_penalty = min(30, (path_length - 3) * 10)
        cost_score -= cost_penalty * 0.3
    
    # Factor 3: New infrastructure (20% weight)
    if path_metrics.get('shortest_path_length') is None:
        # New connection might need new infrastructure
        cost_score -= 20 * 0.2
    
    return max(0, cost_score)


def calculate_roi(revenue_potential: float, cost_score: float) -> float:
    """
    Calculate Return on Investment score.
    
    Args:
        revenue_potential: Revenue potential score (0-100)
        cost_score: Cost score (0-100, higher = lower cost)
    
    Returns:
        ROI score (0-100)
    """
    # Simple ROI: revenue potential vs cost
    # Higher revenue + lower cost = higher ROI
    cost_factor = cost_score / 100.0
    roi = revenue_potential * cost_factor
    return min(100, roi)


def evaluate_prediction(
    prediction: Tuple[str, str, float],
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    G: nx.DiGraph
) -> Dict:
    """
    Evaluate a single predicted link.
    
    Args:
        prediction: Tuple of (source_uri, target_uri, probability)
        nodes_df: DataFrame with node information
        edges_df: DataFrame with edge information
        G: NetworkX graph
    
    Returns:
        Evaluation dictionary with recommendation and metrics
    """
    source_uri, target_uri, probability = prediction
    
    # Get node information
    source_node = nodes_df[nodes_df['uri'] == source_uri].iloc[0].to_dict()
    target_node = nodes_df[nodes_df['uri'] == target_uri].iloc[0].to_dict()
    
    # Calculate path efficiency
    path_metrics = calculate_path_efficiency(G, source_uri, target_uri)
    
    # Estimate revenue potential
    revenue_potential = estimate_revenue_potential(source_node, target_node, path_metrics)
    
    # Estimate implementation cost
    cost_score = estimate_implementation_cost(source_node, target_node, path_metrics)
    
    # Calculate ROI
    roi = calculate_roi(revenue_potential, cost_score)
    
    # Determine recommendation
    if roi >= 70 and revenue_potential >= 60:
        recommendation = 'IMPLEMENT'
    elif roi >= 50 and revenue_potential >= 40:
        recommendation = 'REVIEW'
    else:
        recommendation = 'REJECT'
    
    # Build justification
    justifications = []
    if revenue_potential >= 60:
        justifications.append(f"High revenue potential ({revenue_potential:.1f}/100)")
    if path_metrics.get('path_efficiency_gain', 0) > 0:
        justifications.append(f"Improves path efficiency by {path_metrics['path_efficiency_gain']} hops")
    if cost_score >= 70:
        justifications.append(f"Low implementation cost ({cost_score:.1f}/100)")
    if source_node.get('capacity', 0) - source_node.get('throughput', 0) > 0:
        available = source_node['capacity'] - source_node['throughput']
        justifications.append(f"Available capacity: {available:,.0f}")
    
    justification = "; ".join(justifications) if justifications else "Limited business value"
    
    return {
        'prediction': {
            'source_uri': source_uri,
            'target_uri': target_uri,
            'probability': probability
        },
        'recommendation': recommendation,
        'roi_score': roi,
        'revenue_potential': revenue_potential,
        'cost_score': cost_score,
        'justification': justification,
        'metrics': {
            'available_capacity': max(0, source_node.get('capacity', 0) - source_node.get('throughput', 0)),
            'capacity_utilization': source_node.get('throughput', 0) / source_node.get('capacity', 1) if source_node.get('capacity', 0) > 0 else 0,
            'path_efficiency_gain': path_metrics.get('path_efficiency_gain', 0),
            'shortest_path_length': path_metrics.get('shortest_path_length'),
            'source_type': source_node.get('type', ''),
            'target_type': target_node.get('type', '')
        }
    }


def evaluate_all_predictions(
    predictions: List[Tuple[str, str, float]],
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame
) -> List[Dict]:
    """
    Evaluate all predicted links.
    
    Args:
        predictions: List of prediction tuples
        nodes_df: DataFrame with node information
        edges_df: DataFrame with edge information
    
    Returns:
        List of evaluation dictionaries
    """
    # Build NetworkX graph for path analysis
    G = build_networkx_graph(nodes_df, edges_df)
    
    evaluations = []
    for prediction in predictions:
        try:
            evaluation = evaluate_prediction(prediction, nodes_df, edges_df, G)
            evaluations.append(evaluation)
        except Exception as e:
            print(f"Error evaluating prediction {prediction}: {e}")
            continue
    
    # Sort by ROI score (highest first)
    evaluations.sort(key=lambda x: x['roi_score'], reverse=True)
    
    return evaluations


def write_evaluations_to_stardog(
    conn: stardog.Connection,
    evaluations: List[Dict]
):
    """
    Write evaluation results to Stardog as metadata in a separate graph.
    
    Args:
        conn: Stardog connection
        evaluations: List of evaluation dictionaries
    """
    if not evaluations:
        print("No evaluations to write")
        return
    
    print(f"Writing {len(evaluations)} evaluation results to Stardog...")
    
    # Build SPARQL INSERT query for evaluation metadata
    triples = []
    for eval_result in evaluations:
        pred = eval_result['prediction']
        source_uri = pred['source_uri']
        target_uri = pred['target_uri']
        
        # Create evaluation URI
        eval_uri = f"<urn:Graph_Neural_Network:Evaluation:{source_uri.split(':')[-1]}_{target_uri.split(':')[-1]}>"
        
        # Add evaluation metadata
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:evaluates> <{source_uri}> .")
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:evaluates> <{target_uri}> .")
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:recommendation> \"{eval_result['recommendation']}\" .")
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:roiScore> {eval_result['roi_score']:.2f} .")
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:revenuePotential> {eval_result['revenue_potential']:.2f} .")
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:costScore> {eval_result['cost_score']:.2f} .")
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:justification> \"{eval_result['justification']}\" .")
        triples.append(f"{eval_uri} <urn:Graph_Neural_Network:predictionProbability> {pred['probability']:.4f} .")
    
    insert_query = f"""
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    INSERT DATA {{
        GRAPH <urn:evaluations> {{
            {' '.join(triples)}
        }}
    }}
    """
    
    try:
        conn.begin()
        conn.update(insert_query)
        conn.commit()
        print(f"Successfully wrote {len(evaluations)} evaluations to <urn:evaluations> graph!")
    except Exception as e:
        conn.rollback()
        print(f"Error writing evaluations: {e}")
        raise


if __name__ == '__main__':
    # Example usage
    from predict import predict_missing_links
    from model import LinkPredictionModel
    import torch
    
    # Load data
    print("Loading data...")
    data = load_from_csv('../data/nodes_from_stardog.csv', '../data/edges_from_stardog.csv')
    nodes_df = pd.read_csv('../data/nodes_from_stardog.csv')
    edges_df = pd.read_csv('../data/edges_from_stardog.csv')
    
    # Load model and get predictions
    print("Generating predictions...")
    input_dim = data.x.shape[1]
    model = LinkPredictionModel(input_dim=input_dim, embedding_dim=64, hidden_dim=64, num_layers=2)
    model.load_state_dict(torch.load('../data/model.pt', map_location='cpu'))
    
    predictions = predict_missing_links(model, data, top_k=10)
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    evaluations = evaluate_all_predictions(predictions, nodes_df, edges_df)
    
    # Display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for i, eval_result in enumerate(evaluations, 1):
        pred = eval_result['prediction']
        source_name = pred['source_uri'].split(':')[-1]
        target_name = pred['target_uri'].split(':')[-1]
        
        print(f"\n{i}. {source_name} → {target_name}")
        print(f"   Recommendation: {eval_result['recommendation']}")
        print(f"   ROI Score: {eval_result['roi_score']:.1f}/100")
        print(f"   Revenue Potential: {eval_result['revenue_potential']:.1f}/100")
        print(f"   Cost Score: {eval_result['cost_score']:.1f}/100")
        print(f"   Justification: {eval_result['justification']}")
        print(f"   Prediction Confidence: {pred['probability']:.1%}")
    
    # Write to Stardog
    print("\n" + "="*80)
    ENDPOINT = 'https://presales.stardog.cloud:5820'
    USERNAME = 'aramcodemo2'
    PASSWORD = 'stardog'
    DATABASE = 'JF-gnn'
    
    conn = get_stardog_connection(ENDPOINT, USERNAME, PASSWORD, DATABASE)
    write_evaluations_to_stardog(conn, evaluations)
    conn.close()
    
    print("\nDone!")

