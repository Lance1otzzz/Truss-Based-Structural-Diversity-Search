import time
import numpy as np
from collections import deque

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
    Compute the truss-based structural diversity for each node in G:
    对 G 中的每个节点 v，提取其 1-hop ego 网络，做 k-truss (threshold=k-2) 分解，
    剩余子图的连通分量数即 score[v]。
    """
    n = G.vertex_num
    scores = np.zeros(n, dtype=int)
    # k-truss 的 support 阈值
    t = k - 2

    for v in range(n):
        neigh = G.adj_list[v]
        # 如果邻居数少于 k，则该 ego 网络中不可能有 k-truss
        if len(neigh) < k:
            scores[v] = 0
            continue

        # 构造 induced subgraph: nodes = set(neigh)，edges 只保 u<v 的无向边
        nodes = set(neigh)
        # 邻接表（子图内部）
        adj = {u: set() for u in nodes}
        edges = set()
        for u in nodes:
            for w in G.adj_list[u]:
                if w in nodes:
                    adj[u].add(w)
                    # 只记录一次 (u,w) with u<w
                    if u < w:
                        edges.add((u, w))

        # 如果子图边都没有，直接 0
        if not edges:
            scores[v] = 0
            continue

        # 计算初始 support
        sup = {}
        for (u, w) in edges:
            # triangle count = |N(u) ∩ N(w)|
            common = adj[u].intersection(adj[w])
            sup[(u, w)] = len(common)

        # 初始化待删除队列：support < t 的边
        queue = deque(e for e in edges if sup[e] < t)
        # 剩余的边集合
        remain = set(edges)

        # 迭代删除
        while queue:
            u, w = queue.popleft()
            if (u, w) not in remain:
                continue
            # 删除这条边
            remain.remove((u, w))
            # 在邻接表中也删掉
            adj[u].remove(w)
            adj[w].remove(u)
            # 更新与这条边共三角形的其它两条边的 support
            cset = adj[u].intersection(adj[w])
            for x in cset:
                # 找到它们在 remain 中对应的无向边键
                e1 = (u, x) if u < x else (x, u)
                e2 = (w, x) if w < x else (x, w)
                if e1 in remain:
                    sup[e1] -= 1
                    if sup[e1] == t - 1:
                        queue.append(e1)
                if e2 in remain:
                    sup[e2] -= 1
                    if sup[e2] == t - 1:
                        queue.append(e2)

        # 剩下的 remain 构成 k-truss，统计其连通分量数量
        if not remain:
            scores[v] = 0
        else:
            # 建一个临时的邻接表
            adj_r = {}
            for (u, w) in remain:
                adj_r.setdefault(u, []).append(w)
                adj_r.setdefault(w, []).append(u)
            visited = set()
            comp_cnt = 0
            for u in adj_r:
                if u not in visited:
                    stack = [u]
                    visited.add(u)
                    while stack:
                        x = stack.pop()
                        for y in adj_r[x]:
                            if y not in visited:
                                visited.add(y)
                                stack.append(y)
                    comp_cnt += 1
            scores[v] = comp_cnt

    return scores


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

