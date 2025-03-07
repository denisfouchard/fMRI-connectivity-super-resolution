from MatrixVectorizer import MatrixVectorizer

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj


def evaluation_metrics(pred, true, print: bool = False):

    mae_bc = []
    mae_ec = []
    mae_pc = []

    pred_1d_list = []
    gt_1d_list = []

    num_test_samples = len(pred)
    # Iterate over each test sample
    for i in tqdm(range(num_test_samples)):
        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(pred[i], edge_attr="weight")
        gt_graph = nx.from_numpy_array(true[i], edge_attr="weight")

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")

        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(pred[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(true[i]))

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    # Compute metrics
    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    if print:
        print("MAE: ", mae)
        print("PCC: ", pcc)
        print("Jensen-Shannon Distance: ", js_dis)
        print("Average MAE betweenness centrality:", avg_mae_bc)
        print("Average MAE eigenvector centrality:", avg_mae_ec)
        print("Average MAE PageRank centrality:", avg_mae_pc)

    return {
        "mae": mae,
        "pcc": pcc,
        "js_dis": js_dis,
        "avg_mae_bc": avg_mae_bc,
        "avg_mae_ec": avg_mae_ec,
        "avg_mae_pc": avg_mae_pc,
    }


@torch.no_grad()
def evaluate_model(model, dataloader):
    """
    Runs forward pass, calculates binary predictions (threshold=0.5),
    and returns the accuracy score.
    """
    model.eval()

    preds = []
    true = []
    for batch, target_batch in dataloader:
        batch = batch.to(model.device)
        targets = to_dense_adj(
            target_batch.edge_index,
            edge_attr=target_batch.edge_attr,
            batch=target_batch.batch,
        )

        outputs = model(batch)

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        for i in range(len(outputs)):
            preds.append(outputs[i])
            true.append(targets[i])

    preds = np.array(preds)
    true = np.array(true)

    return np.mean(np.abs(preds - true))
