from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, entropy
from scipy.spatial.distance import jensenshannon
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from utils.matrix_vectorizer import MatrixVectorizer


def print_metrics(gt_matrices, pred_matrices, fold_i):


    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []
    mae_cp = []  # Core-Periphery Structure
    pred_1d_list = []
    gt_1d_list = []
    kl_div_weights = []  # KL divergence for weight distributions

    # Iterate over each test sample
    for i in tqdm(range(len(gt_matrices))):
        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(pred_matrices[i], edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_matrices[i], edge_attr="weight")
        pred_graph.remove_edges_from(nx.selfloop_edges(pred_graph))
        gt_graph.remove_edges_from(nx.selfloop_edges(gt_graph))

        # Extract weights from the adjacency matrices
        gt_weights = [data['weight'] for _, _, data in gt_graph.edges(data=True)]
        pred_weights = [data['weight'] for _, _, data in pred_graph.edges(data=True)]
        
        # If there are no edges in either graph, use placeholder values
        if not gt_weights:
            gt_weights = [0]
        if not pred_weights:
            pred_weights = [0]

        # Create histograms for the weights to get probability distributions
        bins = 50  # Number of bins for histograms
        min_val = min(min(gt_weights), min(pred_weights))
        max_val = max(max(gt_weights), max(pred_weights))
        
        # Create histograms with the same bins for both distributions
        gt_hist, bin_edges = np.histogram(gt_weights, bins=bins, range=(min_val, max_val), density=True)
        pred_hist, _ = np.histogram(pred_weights, bins=bins, range=(min_val, max_val), density=True)
        
        # Add small epsilon to avoid division by zero in KL divergence
        epsilon = 1e-10
        gt_hist = gt_hist + epsilon
        pred_hist = pred_hist + epsilon
        
        # Normalize to ensure they are proper probability distributions
        gt_hist = gt_hist / np.sum(gt_hist)
        pred_hist = pred_hist / np.sum(pred_hist)
        
        # Compute KL divergence between weight distributions
        kl_div = entropy(gt_hist, pred_hist)
        kl_div_weights.append(kl_div)

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight", k=min(10, len(pred_graph.nodes())))
        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight", k=min(10, len(gt_graph.nodes())))

        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight", max_iter=1000)
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight", max_iter=1000)

        pred_pc = nx.pagerank(pred_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")
        
        pred_cp = compute_weighted_kcore(pred_graph)
        gt_cp = compute_weighted_kcore(gt_graph)
        
        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        gt_bc_values = list(gt_bc.values())
        
        pred_ec_values = list(pred_ec.values())
        gt_ec_values = list(gt_ec.values())
        
        pred_pc_values = list(pred_pc.values())
        gt_pc_values = list(gt_pc.values())
        
        pred_cp_values = list(pred_cp.values())
        gt_cp_values = list(gt_cp.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))
        mae_cp.append(mean_absolute_error(pred_cp_values, gt_cp_values))
        pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrices[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt_matrices[i]))

    # Compute average MAEs and KL divergence
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)
    avg_mae_cp = sum(mae_cp) / len(mae_cp)
    avg_kl_div_weights = sum(kl_div_weights) / len(kl_div_weights)

    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)
    
    # Compute metrics
    mae = mean_absolute_error(gt_1d, pred_1d)
    pcc = pearsonr(gt_1d, pred_1d)[0]
    js_dis = jensenshannon(gt_1d, pred_1d)

    print("MAE: ", mae)
    print("PCC: ", pcc)
    print("Jensen-Shannon Distance: ", js_dis)
    print("Average KL Divergence on weight distributions:", avg_kl_div_weights)
    print("Average MAE betweenness centrality:", avg_mae_bc)
    print("Average MAE eigenvector centrality:", avg_mae_ec)
    print("Average MAE PageRank centrality:", avg_mae_pc)
    print("Average MAE core-periphery structure:", avg_mae_cp)
    # Write results to file
    with open(f"results_fold_{i}.txt", "w") as f:
        f.write("MAE: " + str(mae) + "\n")
        f.write("PCC: " + str(pcc) + "\n")
        f.write("Jensen-Shannon Distance: " + str(js_dis) + "\n")
        f.write("Average KL Divergence on weight distributions: " + str(avg_kl_div_weights) + "\n")
        f.write("Average MAE betweenness centrality: " + str(avg_mae_bc) + "\n")
        f.write("Average MAE eigenvector centrality: " + str(avg_mae_ec) + "\n")
        f.write("Average MAE PageRank centrality: " + str(avg_mae_pc) + "\n")
        f.write("Average MAE core-periphery structure: " + str(avg_mae_cp) + "\n")


def compute_weighted_kcore(graph):
    """
    Compute core-periphery structure using NetworkX's k-core decomposition
    adapted for weighted graphs.
    
    Parameters:
    graph (nx.Graph): NetworkX graph with weight attributes on edges
    
    Returns:
    dict: Dictionary mapping nodes to their core-periphery scores
    """
    # Create a new graph for k-core calculation
    # Scale weights to integers for edge multiplicity
    G_multi = nx.Graph()
    
    # Add all nodes
    G_multi.add_nodes_from(graph.nodes())
    
    # Find minimum weight to use for scaling
    all_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    if not all_weights:
        # Return zeros if there are no edges
        return {node: 0 for node in graph.nodes()}
    
    min_weight = min(all_weights)
    scale_factor = 1.0 / min_weight if min_weight > 0 else 1.0
    
    # Add weighted edges as multiple edges
    for u, v, data in graph.edges(data=True):
        # Scale up weight and convert to integer for multiplicity
        weight = max(1, int(data['weight'] * scale_factor))
        G_multi.add_edge(u, v, weight=weight)
    
    # Find maximum k-core using NetworkX function
    k_core = nx.core_number(G_multi)
    
    # Normalize to [0,1] range
    max_core = max(k_core.values()) if k_core.values() else 1
    normalized_cores = {node: core/max_core for node, core in k_core.items()}
    
    return normalized_cores

def evaluate_metrics(model, data_loader):
    gt_matrices = None
    pred_matrices = None
    with torch.no_grad():
        for batch, target_batch in data_loader:
            batch = batch.to(model.device)
            target_batch = target_batch.to(model.device)

            # Forward pass on training data
            outputs = model(batch)

            # Assuming y contains the target adjacency information
            targets = to_dense_adj(target_batch.edge_index, edge_attr=target_batch.edge_attr, batch=target_batch.batch)
            if gt_matrices is None:
                gt_matrices = targets.detach().cpu().numpy()
                pred_matrices = outputs.detach().cpu().numpy()
            else:
                gt_matrices = np.concat((gt_matrices,targets.detach().cpu().numpy()), axis=0)
                pred_matrices = np.concat((pred_matrices,outputs.detach().cpu().numpy()), axis=0)
    
    print_metrics(gt_matrices, pred_matrices)

            

