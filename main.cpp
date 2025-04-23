
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
using namespace std;
using namespace std::chrono;

class UndirectedGraph {
public:
    int vertex_num;
    vector<vector<int>> adj_list;

    UndirectedGraph(const vector<pair<int, int>>& edge_list) {
        vertex_num = 0;
        for (auto& edge : edge_list) {
            vertex_num = max(vertex_num, max(edge.first, edge.second) + 1);
        }
        adj_list.resize(vertex_num);
        for (auto& edge : edge_list) {
            int u = edge.first;
            int v = edge.second;
            if (u == v) continue; // skip self-loops
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
    }
};

bool check_result(const vector<int>& prediction, const vector<int>& ground_truth) {
    return prediction == ground_truth;
}

vector<pair<int, int>> load_edge_list(const string& filename) {
    vector<pair<int, int>> edges;
    ifstream infile(filename);
    int u, v;
    while (infile >> u >> v) {
        if (u == v) continue;
        if (u > v) swap(u, v);
        edges.emplace_back(u, v);
    }
    infile.close();
    sort(edges.begin(), edges.end());
    edges.erase(unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

vector<int> load_ground_truth(const string& filename) {
    vector<int> result;
    ifstream infile(filename);
    int val;
    while (infile >> val) {
        result.push_back(val);
    }
    infile.close();
    return result;
}

vector<int> compute_structure_diversity(const UndirectedGraph& G, int k) {
    // ############################################################################
    // # Structure Diversity Calculation
    // ############################################################################
    vector<int> diversity(G.vertex_num, 0);
    return diversity;
}

int main() {
    cout << "##### Loading the dataset..." << endl;

    vector<pair<int, int>> edge_list = load_edge_list("graph.txt");
    UndirectedGraph G(edge_list);

    vector<vector<int>> ground_truths;
    ground_truths.push_back(load_ground_truth("results(k=3).txt"));
    ground_truths.push_back(load_ground_truth("results(k=4).txt"));
    ground_truths.push_back(load_ground_truth("results(k=5).txt"));
    ground_truths.push_back(load_ground_truth("results(k=6).txt"));

    vector<vector<int>> connectivity_results;

    cout << "##### Test ..." << endl;
    auto start = high_resolution_clock::now();
    connectivity_results.push_back(compute_structure_diversity(G, 3));
    connectivity_results.push_back(compute_structure_diversity(G, 4));
    connectivity_results.push_back(compute_structure_diversity(G, 5));
    connectivity_results.push_back(compute_structure_diversity(G, 6));
    auto end = high_resolution_clock::now();
    double duration = duration_cast<milliseconds>(end - start).count() / 1000.0;

    cout << "Processing time: " << duration << " seconds" << endl;
    for (int i = 0; i < 4; ++i) {
        if (check_result(connectivity_results[i], ground_truths[i])) {
            cout << "Correct result(k=" << (i + 3) << ")" << endl;
        } else {
            cout << "Incorrect result(k=" << (i + 3) << ")" << endl;
        }
    }

    return 0;
}
