import time
import numpy as np

############################################################################
# Graph definition and utility
############################################################################

class UndirectedGraph(object):
    def __init__(self, edge_list):
        self.vertex_num = np.max(edge_list[:, :2]) + 1
        self.adj_list = [[] for _ in range(self.vertex_num)]

        for src, dst in edge_list:
            self.adj_list[src].append(dst)
            self.adj_list[dst].append(src)

def check_result(prediction, ground_truth):
    return np.array_equal(prediction, ground_truth)


############################################################################
# Structure Diversity Calculation
############################################################################

def compute_structure_diversity(G, k):
    """
    Compute the structure diversity for each node in the graph G,
    defined as the number of maximal connected k-truss subgraphs
    in its 1-hop ego network.

    Args:
        G: An instance of UndirectedGraph
        k: k-truss parameter

    Returns:
        A numpy array: diversity score for each node
    """
    diversity_scores = []

    return np.array(diversity_scores, dtype=int)


############################################################################
# Main Execution
############################################################################

if __name__ == "__main__":
    print('\n##### Loading the dataset...')
    edge_list = np.loadtxt('graph.txt', dtype=int)
    if edge_list.ndim == 1:
        edge_list = edge_list.reshape(1, -1)
    ground_truths = []
    ground_truths.append(np.loadtxt('results(k=3).txt', dtype=int))
    ground_truths.append(np.loadtxt('results(k=4).txt', dtype=int))
    ground_truths.append(np.loadtxt('results(k=5).txt', dtype=int))
    ground_truths.append(np.loadtxt('results(k=6).txt', dtype=int))

    connectivity_results = []
    G = UndirectedGraph(edge_list)

    print('\n##### Test ...')
    start = time.time()
    connectivity_results.append(compute_structure_diversity(G, 3))
    connectivity_results.append(compute_structure_diversity(G, 4))
    connectivity_results.append(compute_structure_diversity(G, 5))
    connectivity_results.append(compute_structure_diversity(G, 6))
    end = time.time()
    print("Processing time: {}".format(end - start))
    print("Correct result(k=3)" if check_result(connectivity_results[0], ground_truths[0]) else "Incorrect result(k=3)")
    print("Correct result(k=4)" if check_result(connectivity_results[1], ground_truths[1]) else "Incorrect result(k=4)")
    print("Correct result(k=5)" if check_result(connectivity_results[2], ground_truths[2]) else "Incorrect result(k=5)")
    print("Correct result(k=6)" if check_result(connectivity_results[3], ground_truths[3]) else "Incorrect result(k=6)")

