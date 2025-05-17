import os
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional

import numpy as np



###############################################################################
# 1. Lightweight undirected graph                                             #
###############################################################################

class UndirectedGraph:
    """Adjacency‑list undirected graph with 0‑based integer vertices."""

    def __init__(self, edge_list: np.ndarray):
        if edge_list.size == 0:
            self.vertex_num = 0
            self.adj_list: List[List[int]] = []
            return
        edge_list = edge_list.reshape(-1, 2)
        self.vertex_num = int(edge_list[:, :2].max()) + 1
        self.adj_list = [[] for _ in range(self.vertex_num)]
        for src, dst in edge_list:
            if src == dst:
                continue  # ignore self‑loops
            self.adj_list[src].append(dst)
            self.adj_list[dst].append(src)
        # sort & dedup
        for v in range(self.vertex_num):
            self.adj_list[v] = sorted(set(self.adj_list[v]))

###############################################################################
# 2. Validation helper                                                        #
###############################################################################

def check_result(prediction: np.ndarray, ground_truth: np.ndarray) -> bool:
    return np.array_equal(prediction, ground_truth)



###############################################################################
# 4. GCT‑Index                                                                #
###############################################################################

class GCTIndex:
    """Global‑Context‑Truss index (Algorithms 5‑6) with optional acceleration."""

    def __init__(self, parallel: bool | int = False):
        if isinstance(parallel, bool):
            self._n_jobs = os.cpu_count() if parallel else 1
        else:
            self._n_jobs = max(1, parallel)
        self.index: Dict[int, Dict] = {}
        self._score_cache = {}  # 缓存计算结果

    # ------------------------------------------------------------------
    def build_index(self, G: UndirectedGraph):
        if G.vertex_num == 0:
            return

        # 4.1  Triangle enumeration ------------------------------------------
        t_enum_start = time.time()
        triangle_edges: List[List[Tuple[int, int]]]
        triangle_edges = _enum_triangles_python(G)
        t_enum = time.time() - t_enum_start

        # 4.2  Build each vertex's GCT (parallel) ------------------------------
        t_build_start = time.time()
        if self._n_jobs == 1 or G.vertex_num < 1000:
            for v in range(G.vertex_num):
                self.index[v] = _build_single_vertex(v, triangle_edges[v])
        else:
            with ProcessPoolExecutor(max_workers=self._n_jobs) as pool:
                fut_to_v = {pool.submit(_build_single_vertex, v, triangle_edges[v]): v for v in range(G.vertex_num)}
                for fut in as_completed(fut_to_v):
                    v, idx_entry = fut.result()
                    self.index[v] = idx_entry
        t_build = time.time() - t_build_start
        print(f"    Triangle enumeration: {t_enum:.3f}s  (Python)  |  GCT build: {t_build:.3f}s  ({self._n_jobs} worker(s))")

    # ------------------------------------------------------------------
    def compute_score(self, v: int, k: int) -> int:
        """计算节点v在k-truss约束下的连通分量数"""
        # 1. 使用缓存避免重复计算
        cache_key = (v, k)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
            
        # 2. 获取索引数据
        idx = self.index.get(v)
        if not idx or not idx["supernodes"]:
            self._score_cache[cache_key] = 0
            return 0
            
        # 3. 筛选trussness >= k的超节点
        valid_supernodes = {}
        for sn, data in idx["supernodes"].items():
            if data["trussness"] >= k:
                valid_supernodes[sn] = data
                
        if not valid_supernodes:
            self._score_cache[cache_key] = 0
            return 0
            
        # 4. 构建超节点间的邻接表（只保留trussness >= k的超边）
        adj_list = defaultdict(list)
        for s, t, tr in idx["superedges"]:
            if tr >= k and s in valid_supernodes and t in valid_supernodes:
                adj_list[s].append(t)
                adj_list[t].append(s)
                
        # 5. 计算连通分量数
        visited = set()
        cc_count = 0
        
        for node in valid_supernodes:
            if node in visited:
                continue
                
            cc_count += 1
            queue = deque([node])
            visited.add(node)
            
            while queue:
                curr = queue.popleft()
                for neighbor in adj_list[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # 6. 缓存并返回结果
        self._score_cache[cache_key] = cc_count
        return cc_count

###############################################################################
# Helper: pure‑Python triangle enumeration (fallback)                         #
###############################################################################

def _enum_triangles_python(G: UndirectedGraph) -> List[List[Tuple[int, int]]]:
    neighbour_sets = [set(nbrs) for nbrs in G.adj_list]
    tri_edges: List[List[Tuple[int, int]]] = [[] for _ in range(G.vertex_num)]
    for w in range(G.vertex_num):
        nbrs = G.adj_list[w]
        ln = len(nbrs)
        for i in range(ln):
            u = nbrs[i]
            for j in range(i + 1, ln):
                v = nbrs[j]
                if v in neighbour_sets[u]:
                    tri_edges[w].append((u, v) if u < v else (v, u))
    return tri_edges

###############################################################################
# 5. Worker helpers (serial per vertex)                                       #
###############################################################################

def _build_single_vertex(v: int, tri_edges_v: List[Tuple[int, int]]):
    edges = list(set(tri_edges_v))
    if not edges:
        return v, {"supernodes": {}, "superedges": []}
    verts: List[int] = sorted({x for e in edges for x in e})
    adj_ego: Dict[int, Set[int]] = defaultdict(set)
    for a, b in edges:
        adj_ego[a].add(b)
        adj_ego[b].add(a)
    support = {(a, b): len(adj_ego[a].intersection(adj_ego[b])) for a, b in edges}
    edge_truss = _truss_decomposition_serial(verts, edges, support)
    idx_entry = _construct_gct_serial(verts, edge_truss)
    return v, idx_entry

###############################################################################
# 6. Serial truss decomposition & GCT construction (unchanged)                #
###############################################################################

# (Functions _truss_decomposition_serial and _construct_gct_serial remain
# identical to previous revision; omitted here for brevity.)

from math import inf  # noqa: E402 – needed for type hints later

# -- truss decomposition ------------------------------------------------------

def _truss_decomposition_serial(vertices: List[int], edges: List[Tuple[int, int]],
                                support: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    if not edges:
        return {}
    remaining: Set[Tuple[int, int]] = set(edges)
    sup = support.copy()
    max_sup = max(sup.values()) if sup else 0
    buckets: List[List[Tuple[int, int]]] = [[] for _ in range(max_sup + 1)]
    edge_bucket: Dict[Tuple[int, int], int] = {}
    for e, s in sup.items():
        buckets[s].append(e)
        edge_bucket[e] = s
    adj: Dict[int, Set[int]] = defaultdict(set)
    for u, w in edges:
        adj[u].add(w)
        adj[w].add(u)
    edge_truss: Dict[Tuple[int, int], int] = {}
    for k in range(max_sup + 1):
        while buckets[k]:
            e = buckets[k].pop()
            if e not in remaining:
                continue
            remaining.remove(e)
            edge_truss[e] = k + 2
            u, w = e
            common = adj[u].intersection(adj[w])
            for x in common:
                e1 = (min(u, x), max(u, x))
                e2 = (min(w, x), max(w, x))
                if e1 in remaining and e2 in remaining:
                    for affected in (e1, e2):
                        old_sup = edge_bucket[affected]
                        if old_sup > k:
                            buckets[old_sup].remove(affected)
                            new_sup = old_sup - 1
                            buckets[new_sup].append(affected)
                            edge_bucket[affected] = new_sup
                            sup[affected] = new_sup
    return edge_truss

# -- GCT construction ---------------------------------------------------------

def _construct_gct_serial(vertices: List[int], edge_trussness: Dict[Tuple[int, int], int]):
    V_super: Dict[int, Dict] = {u: {"trussness": 2, "vertices": {u}} for u in vertices}
    for (u, w), tr in edge_trussness.items():
        V_super[u]["trussness"] = max(V_super[u]["trussness"], tr)
        V_super[w]["trussness"] = max(V_super[w]["trussness"], tr)
    parent = {u: u for u in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    E_super: List[Tuple[int, int, int]] = []
    for (u, w), tr in sorted(edge_trussness.items(), key=lambda x: x[1], reverse=True):
        ru, rw = find(u), find(w)
        if ru == rw:
            continue
        if V_super[ru]["trussness"] == V_super[rw]["trussness"] == tr:
            union(ru, rw)
            r = find(ru)
            other = rw if r == ru else ru
            V_super[r]["vertices"].update(V_super[other]["vertices"])
            new_E: List[Tuple[int, int, int]] = []
            for s, t, wt in E_super:
                s2 = r if s == other else s
                t2 = r if t == other else t
                if s2 != t2:
                    new_E.append((s2, t2, wt))
            E_super = new_E
            del V_super[other]
        else:
            E_super.append((ru, rw, tr))
    return {"supernodes": V_super, "superedges": E_super}

###############################################################################
# 7. Main – evaluation pipeline unchanged                                     #
###############################################################################

if __name__ == "__main__":
    print("\n##### Loading the dataset…\n")
    try:
        edge_list = np.loadtxt("graph.txt", dtype=int)
    except OSError:
        print("Error: graph.txt not found. Please create it with the edge list.")
        edge_list = np.array([])
    if edge_list.size == 0:
        G = UndirectedGraph(np.array([]))
    else:
        if edge_list.ndim == 1 and edge_list.shape[0] == 2:
            edge_list = edge_list.reshape(1, 2)
        elif edge_list.ndim == 0:
            print("graph.txt contains a scalar, invalid.")
            edge_list = np.array([])
        G = UndirectedGraph(edge_list)

    # Load ground‑truth files ----------------------------------------------------
    ground_truths: List[Optional[np.ndarray]] = []
    for k_gt in range(3, 7):
        try:
            ground_truths.append(np.loadtxt(f"results(k={k_gt}).txt", dtype=int))
        except (OSError, ValueError):
            ground_truths.append(None)

    print("\n##### Building GCT‑Index")
    gct = GCTIndex(parallel=True)
    if G.vertex_num > 0:
        t0 = time.time()
        gct.build_index(G)
        print(f"Total index build time: {time.time() - t0:.3f}s")

        print("\n##### Test with GCT‑Index method…\n")
        gct_results: List[np.ndarray] = []
        t1 = time.time()
        for k_test in range(3, 7):
            scores = np.fromiter((gct.compute_score(v, k_test) for v in range(G.vertex_num)),
                                 dtype=np.int64, count=G.vertex_num)
            gct_results.append(scores)
        print(f"GCT query processing time: {time.time() - t1:.3f}s")

        for idx, k_val in enumerate(range(3, 7)):
            gt = ground_truths[idx]
            pred = gct_results[idx]
            if gt is None:
                print(f"GCT result (k={k_val}) (no ground truth): {pred}")
            elif pred.shape != gt.shape:
                print(f"GCT incorrect (k={k_val}): shape mismatch (pred {pred.shape} vs gt {gt.shape})")
            else:
                print(f"GCT correct (k={k_val}): {check_result(pred, gt)}")
    else:
        print("Graph is empty – skipping GCT‑Index build and test.")
