import time
import numpy as np
from collections import deque

############################################################################
# Graph definition
############################################################################

class UndirectedGraph(object):
    def __init__(self, edge_list):
        self.vertex_num = int(np.max(edge_list[:, :2]) + 1)
        self.adj_list = [[] for _ in range(self.vertex_num)]
        for u, v in edge_list:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

def check_result(pred, truth):
    return np.array_equal(pred, truth)


############################################################################
# 版本一：Baseline
#   - 对每个 v 独立提取 ego 网络
#   - 完全局部 support 计算（不借助全局 supG）
#   - 剥离后用 BFS/DFS 计连通分量
############################################################################
def compute_structure_diversity_baseline(G, k):
    n = G.vertex_num
    threshold = k - 2
    scores = np.zeros(n, dtype=int)

    for v in range(n):
        neigh = G.adj_list[v]
        if len(neigh) < k:
            continue

        # 构造诱导子图
        nodes = set(neigh)
        adj = {u: set() for u in nodes}
        edges = []
        for u in nodes:
            for w in G.adj_list[u]:
                if w in nodes and u < w:
                    adj[u].add(w)
                    adj[w].add(u)
                    edges.append((u, w))

        if not edges:
            continue

        # 局部 support
        support = {}
        for (u, w) in edges:
            support[(u, w)] = len(adj[u] & adj[w])

        # k-truss peeling
        q = deque(e for e in edges if support[e] < threshold)
        remain = set(edges)
        while q:
            u, w = q.popleft()
            if (u, w) not in remain:
                continue
            remain.remove((u, w))
            adj[u].remove(w)
            adj[w].remove(u)

            common = adj[u] & adj[w]
            for x in common:
                e1 = (u, x) if u < x else (x, u)
                e2 = (w, x) if w < x else (x, w)
                if e1 in remain:
                    support[e1] -= 1
                    if support[e1] == threshold - 1:
                        q.append(e1)
                if e2 in remain:
                    support[e2] -= 1
                    if support[e2] == threshold - 1:
                        q.append(e2)

        # BFS 计连通分量
        if remain:
            visited = set()
            comp = 0
            # 构造残余子图邻接
            adj_r = {}
            for u, w in remain:
                adj_r.setdefault(u, []).append(w)
                adj_r.setdefault(w, []).append(u)
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
                    comp += 1
            scores[v] = comp
    return scores


############################################################################
# 全局预处理：计算 supG[(u,v)] = |N(u) ∩ N(v)| for u<v
############################################################################
def compute_global_support(G):
    n = G.vertex_num
    adj_sets = [set(nb) for nb in G.adj_list]
    supG = {}
    for u in range(n):
        Nu = adj_sets[u]
        for v in G.adj_list[u]:
            if u < v:
                supG[(u, v)] = len(Nu & adj_sets[v])
    return supG


############################################################################
# 版本二：Global-Prune
#   - 先全局一次性计算 supG
#   - 给每个 v 的 ego 网络只保留 supG >= threshold 的边
#   - 其余完全同版本一，用 BFS 计连通分量
############################################################################
def compute_structure_diversity_global_prune(G, k, supG):
    n = G.vertex_num
    threshold = k - 2
    scores = np.zeros(n, dtype=int)

    for v in range(n):
        neigh = G.adj_list[v]
        if len(neigh) < k:
            continue

        nodes = set(neigh)
        adj = {u: set() for u in nodes}
        edges = []
        # 只保留 supG >= threshold 的边
        for u in nodes:
            for w in G.adj_list[u]:
                if w in nodes and u < w and supG.get((u, w), 0) >= threshold:
                    adj[u].add(w)
                    adj[w].add(u)
                    edges.append((u, w))
        if not edges:
            continue

        # 局部真正 support 计算
        support = {}
        for (u, w) in edges:
            support[(u, w)] = len(adj[u] & adj[w])

        # peeling
        q = deque(e for e in edges if support[e] < threshold)
        remain = set(edges)
        while q:
            u, w = q.popleft()
            if (u, w) not in remain:
                continue
            remain.remove((u, w))
            adj[u].remove(w)
            adj[w].remove(u)

            common = adj[u] & adj[w]
            for x in common:
                e1 = (u, x) if u < x else (x, u)
                e2 = (w, x) if w < x else (x, w)
                if e1 in remain:
                    support[e1] -= 1
                    if support[e1] == threshold - 1:
                        q.append(e1)
                if e2 in remain:
                    support[e2] -= 1
                    if support[e2] == threshold - 1:
                        q.append(e2)

        # BFS 连通分量
        if remain:
            visited = set()
            comp = 0
            adj_r = {}
            for u, w in remain:
                adj_r.setdefault(u, []).append(w)
                adj_r.setdefault(w, []).append(u)
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
                    comp += 1
            scores[v] = comp
    return scores


############################################################################
# 版本三：Global-Prune + Union-Find
#   - 与版本二相同的剥离流程
#   - 用 union-find 计连通分量
############################################################################
def compute_structure_diversity_global_unionfind(G, k, supG):
    n = G.vertex_num
    threshold = k - 2
    scores = np.zeros(n, dtype=int)

    for v in range(n):
        neigh = G.adj_list[v]
        if len(neigh) < k:
            continue

        nodes = set(neigh)
        adj = {u: set() for u in nodes}
        edges = []
        for u in nodes:
            for w in G.adj_list[u]:
                if w in nodes and u < w and supG.get((u, w), 0) >= threshold:
                    adj[u].add(w)
                    adj[w].add(u)
                    edges.append((u, w))
        if not edges:
            continue

        support = {}
        for (u, w) in edges:
            support[(u, w)] = len(adj[u] & adj[w])

        q = deque(e for e in edges if support[e] < threshold)
        remain = set(edges)
        while q:
            u, w = q.popleft()
            if (u, w) not in remain:
                continue
            remain.remove((u, w))
            adj[u].remove(w)
            adj[w].remove(u)
            common = adj[u] & adj[w]
            for x in common:
                e1 = (u, x) if u < x else (x, u)
                e2 = (w, x) if w < x else (x, w)
                if e1 in remain:
                    support[e1] -= 1
                    if support[e1] == threshold - 1:
                        q.append(e1)
                if e2 in remain:
                    support[e2] -= 1
                    if support[e2] == threshold - 1:
                        q.append(e2)

        # Union-Find 计连通分量
        if remain:
            parent = {}
            for u, w in remain:
                parent[u] = u
                parent[w] = w
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for u, w in remain:
                union(u, w)
            roots = set(find(x) for x in parent)
            scores[v] = len(roots)

    return scores


############################################################################
# Main for timing & correctness
############################################################################
if __name__ == "__main__":
    # 读图
    edge_list = np.loadtxt('graph.txt', dtype=int)
    if edge_list.ndim == 1:
        edge_list = edge_list.reshape(1, -1)
    G = UndirectedGraph(edge_list)

    # 读 ground-truth
    ground_truths = [
        np.loadtxt('results(k=3).txt', dtype=int),
        np.loadtxt('results(k=4).txt', dtype=int),
        np.loadtxt('results(k=5).txt', dtype=int),
        np.loadtxt('results(k=6).txt', dtype=int),
    ]

    # 预计算全局 support
    print("Compute global support …")
    t0 = time.time()
    supG = compute_global_support(G)
    print(" global support done in {:.3f}s".format(time.time() - t0))

    # 分别测试三种版本
    for version, func in [
        ("Baseline", compute_structure_diversity_baseline),
        ("Global‑Prune", compute_structure_diversity_global_prune),
        ("Global+UF", compute_structure_diversity_global_unionfind)
    ]:
        print(f"\n=== Running {version} ===")
        t1 = time.time()
        results = []
        for k in [3, 4, 5, 6]:
            if version == "Baseline":
                res = func(G, k)
            else:
                res = func(G, k, supG)
            ok = check_result(res, ground_truths[k - 3])
            print(f" k={k} time so far {(time.time()-t1):.3f}s, correct: {ok}")
        print(f"{version} TOTAL {(time.time()-t1):.3f}s")
