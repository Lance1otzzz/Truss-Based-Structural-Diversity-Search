# file: ego_truss_numba.py
import numpy as np
import time
from numba import njit

def load_csr(edge_list):
    """
    从边列表 (m×2) 构造 CSR：
    - 两个方向都插边
    - 按 row 排序
    返回 n, indptr, nbrs
    """
    # 1) 确定 n
    n = int(edge_list.max() + 1)
    # 2) 构造双向边
    u = edge_list[:,0]
    v = edge_list[:,1]
    all_u = np.empty(u.size*2, dtype=np.int32)
    all_v = np.empty(v.size*2, dtype=np.int32)
    all_u[:u.size] = u;     all_v[:v.size] = v
    all_u[u.size:] = v;     all_v[v.size:] = u
    # 3) 按 u 排序
    idx = np.argsort(all_u)
    rows = all_u[idx]
    cols = all_v[idx]
    # 4) 生成 indptr
    indptr = np.zeros(n+1, dtype=np.int32)
    # count
    np.add.at(indptr, rows+1, 1)
    # prefix sum
    np.cumsum(indptr, out=indptr)
    nbrs = cols
    return n, indptr, nbrs

@njit
def compute_scores_numba(n, indptr, nbrs, k,
                         max_local_e,
                         pos, eu, ev, support, alive, inq, qarr,
                         loc_deg, loc_indptr, loc_nbrs, loc_idx,
                         parent, has_edge):
    """
    n, indptr, nbrs          : 全图 CSR
    k                        : k-truss 参数
    max_local_e             : max_deg*(max_deg-1)//2, ego 子图最大边数
    后面一大堆数组都是预先在 Python 端 malloc 好并传进来的缓冲区，
    避免 JIT 中反复 alloc。
    """
    out = np.zeros(n, dtype=np.int32)
    threshold = k - 2

    for v in range(n):
        # 1) 本节点 ego 节点集 = nbrs[indptr[v]:indptr[v+1]]
        s = indptr[v]; t = indptr[v+1]
        deg = t - s
        if deg < k:
            continue

        # 2) 标记本 ego 节点的 pos（值 = local_index+1；0 表示不在 ego）
        for i in range(s, t):
            pos[nbrs[i]] = i - s + 1

        # 3) 构造本地边列表 (eu, ev)，只保留 u<v
        e_cnt = 0
        for i in range(s, t):
            u_local = i - s
            U = nbrs[i]
            for p in range(indptr[U], indptr[U+1]):
                W = nbrs[p]
                pw = pos[W]
                if pw != 0:
                    w_local = pw - 1
                    if u_local < w_local:
                        eu[e_cnt] = u_local
                        ev[e_cnt] = w_local
                        e_cnt += 1

        if e_cnt == 0:
            # 清理 pos
            for i in range(s, t):
                pos[nbrs[i]] = 0
            continue
        # 若 e_cnt 很大，请保证 max_local_e 足够大

        # 4) 构造本地 CSR：loc_indptr, loc_nbrs, loc_idx
        #    loc_idx 存每条邻接对应的边索引 e in [0,e_cnt)
        # reset deg
        for i in range(deg):
            loc_deg[i] = 0
        for e in range(e_cnt):
            u0 = eu[e]; v0 = ev[e]
            loc_deg[u0] += 1
            loc_deg[v0] += 1
        loc_indptr[0] = 0
        for i in range(deg):
            loc_indptr[i+1] = loc_indptr[i] + loc_deg[i]
        # 暂用 loc_deg 存当前填充指针
        for i in range(deg):
            loc_deg[i] = loc_indptr[i]

        for e in range(e_cnt):
            u0 = eu[e]; v0 = ev[e]
            pu = loc_deg[u0]
            loc_nbrs[pu] = v0
            loc_idx[pu]  = e
            loc_deg[u0]  = pu + 1

            pv = loc_deg[v0]
            loc_nbrs[pv] = u0
            loc_idx[pv]  = e
            loc_deg[v0]  = pv + 1

        # 5) 计算初始 support 和 peeling 队列
        head = 0; tail = 0
        for e in range(e_cnt):
            # triangle count = N(u)∩N(v)
            u0 = eu[e]; v0 = ev[e]
            p1 = loc_indptr[u0]; end1 = loc_indptr[u0+1]
            p2 = loc_indptr[v0]; end2 = loc_indptr[v0+1]
            cnt = 0
            while p1 < end1 and p2 < end2:
                x1 = loc_nbrs[p1]; x2 = loc_nbrs[p2]
                if x1 < x2:
                    p1 += 1
                elif x2 < x1:
                    p2 += 1
                else:
                    cnt += 1; p1 += 1; p2 += 1
            support[e] = cnt
            alive[e] = 1
            inq[e]   = 0
            if cnt < threshold:
                qarr[tail] = e
                tail += 1
                inq[e] = 1

        # 6) k-truss peeling
        while head < tail:
            e0 = qarr[head]; head += 1
            if alive[e0] == 0:
                continue
            alive[e0] = 0
            u0 = eu[e0]; v0 = ev[e0]
            p1 = loc_indptr[u0]; end1 = loc_indptr[u0+1]
            p2 = loc_indptr[v0]; end2 = loc_indptr[v0+1]
            # 找所有共同邻居，更新对应边的 support
            while p1 < end1 and p2 < end2:
                x1 = loc_nbrs[p1]; x2 = loc_nbrs[p2]
                if x1 < x2:
                    p1 += 1
                elif x2 < x1:
                    p2 += 1
                else:
                    # 更新 (u0,x1) 和 (v0,x1) 这两条边的 support
                    e1 = loc_idx[p1]
                    e2 = loc_idx[p2]
                    if alive[e1]:
                        support[e1] -= 1
                        if support[e1] < threshold and inq[e1] == 0:
                            qarr[tail] = e1; tail += 1; inq[e1] = 1
                    if alive[e2]:
                        support[e2] -= 1
                        if support[e2] < threshold and inq[e2] == 0:
                            qarr[tail] = e2; tail += 1; inq[e2] = 1
                    p1 += 1; p2 += 1

        # 7) Union‑Find 统计剩余边的连通分量
        #    只统计那些在 remain 边里出现过的局部节点
        for i in range(deg):
            has_edge[i] = 0
            parent[i]   = i

        for e in range(e_cnt):
            if alive[e]:
                u0 = eu[e]; v0 = ev[e]
                has_edge[u0] = 1
                has_edge[v0] = 1
                # union(u0,v0)
                # find ru
                x = u0
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                ru = x
                y = v0
                while parent[y] != y:
                    parent[y] = parent[parent[y]]; y = parent[y]
                rv = y
                if ru != rv:
                    parent[rv] = ru

        comp = 0
        for i in range(deg):
            if has_edge[i] and parent[i] == i:
                comp += 1
        out[v] = comp

        # 8) 清理 pos
        for i in range(s, t):
            pos[nbrs[i]] = 0

    return out


# 添加 check_result 函数
def check_result(prediction, ground_truth):
    return np.array_equal(prediction, ground_truth)

if __name__ == '__main__':
    # 1) Loading
    print('\n##### Loading the dataset...')
    edges = np.loadtxt('graph.txt', dtype=np.int32)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)

    # 2) Load ground‑truth
    ground_truths = [
        np.loadtxt('results(k=3).txt', dtype=int),
        np.loadtxt('results(k=4).txt', dtype=int),
        np.loadtxt('results(k=5).txt', dtype=int),
        np.loadtxt('results(k=6).txt', dtype=int),
    ]

    # 3) Build graph (CSR)
    print('\n##### Build graph ...')
    n, indptr, nbrs = load_csr(edges)

    # --------- 预分配所有 JIT 缓冲区 -------------
    degs = indptr[1:] - indptr[:-1]
    max_deg = int(degs.max()) if degs.size > 0 else 0 # 处理空图情况
    # 如果 max_deg 为 0，max_local_e 也应为 0 或 1，避免负数
    max_local_e = max(1, max_deg * (max_deg - 1) // 2)

    pos        = np.zeros(n,          dtype=np.int32)
    eu         = np.empty(max_local_e, dtype=np.int32)
    ev         = np.empty(max_local_e, dtype=np.int32)
    support    = np.empty(max_local_e, dtype=np.int32)
    alive      = np.empty(max_local_e, dtype=np.uint8)
    inq        = np.empty(max_local_e, dtype=np.uint8)
    qarr       = np.empty(max_local_e, dtype=np.int32)
    # 确保 loc_deg 等数组大小至少为 1，即使 max_deg 为 0
    loc_deg    = np.empty(max(1, max_deg),    dtype=np.int32)
    loc_indptr = np.empty(max(1, max_deg)+1,  dtype=np.int32)
    loc_nbrs   = np.empty(max(1, max_local_e*2), dtype=np.int32)
    loc_idx    = np.empty(max(1, max_local_e*2), dtype=np.int32)
    parent     = np.empty(max(1, max_deg),    dtype=np.int32)
    has_edge   = np.empty(max(1, max_deg),    dtype=np.uint8)

    # 触发一次 JIT 编译（空跑） - 确保缓冲区大小至少为1
    _ = compute_scores_numba(0, np.array([0], dtype=np.int32), np.array([], dtype=np.int32), 3,
                            max_local_e,
                            pos[:0], eu[:0], ev[:0], support[:0], alive[:0], inq[:0], qarr[:0],
                            loc_deg[:0], loc_indptr[:1], loc_nbrs[:0], loc_idx[:0],
                            parent[:0], has_edge[:0])

    # 4) Test structure diversity
    print('\n##### Testing structure diversity ...')
    start = time.time()
    connectivity_results = []
    for k in [3, 4, 5, 6]:
        # 每次调用前重置 pos (虽然函数内部会清理，但更安全)
        pos.fill(0)
        scores = compute_scores_numba(n, indptr, nbrs, k,
                                      max_local_e,
                                      pos, eu, ev, support, alive, inq, qarr,
                                      loc_deg, loc_indptr, loc_nbrs, loc_idx,
                                      parent, has_edge)
        connectivity_results.append(scores)
    end = time.time()

    # 5) Summary
    print(f"\nProcessing time (all k): {end - start:.3f}s")
    for idx, k in enumerate([3, 4, 5, 6]):
        ok = check_result(connectivity_results[idx], ground_truths[idx])
        print(f"{'Correct' if ok else 'Incorrect'} result (k={k})")
