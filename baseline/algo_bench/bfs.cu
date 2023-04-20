#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>

#include <limits.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <chrono>
using namespace std;

constexpr int THREADS_PER_BLOCK = 1024;


typedef struct {
    int u;
    int v;
    int w;
} Edge;

int next_int (ifstream &ifs) {
    while (!isdigit(ifs.peek())) {
        ifs.get();
    }
    int output;
    ifs >> output;
    return output;
}

Edge* parse_file (const char *filename, int &nodes, int &edges) {
    ifstream ifs(filename);

    nodes = next_int(ifs);
    edges = next_int(ifs);

    Edge *out = (Edge*) malloc(edges*sizeof(Edge));
    for (int i = 0; i < edges; i++) {
        out[i].u = next_int(ifs);
        out[i].v = next_int(ifs);
        out[i].w = next_int(ifs);
    }
    return out;
}

__global__ void init_output (int *output, int num_nodes, int start) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == start) {
        output[index] = 0;
    } else if (index < num_nodes) {
        output[index] = INT_MAX;
    }
}

__global__ void init_bool (bool *value, bool init) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) {
        *value = init;
    }
}

__global__ void bfs (Edge *edge_list, int *output, bool *done, int num_nodes, int num_edges, int iteration) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_edges &&
            output[edge_list[index].u] == iteration &&
            output[edge_list[index].v] == INT_MAX) {
        output[edge_list[index].v] = iteration + 1;
        *done = false;
    }
}

Edge* init_edge_list_cuda (Edge *h_edge_list, int num_edges) {
    Edge *d_edge_list;
    int bytes = num_edges*sizeof(Edge);
    cudaMalloc((void **) &d_edge_list, bytes);
    cudaMemcpy(d_edge_list, h_edge_list, bytes, cudaMemcpyHostToDevice);
    return d_edge_list;
}

int* init_output_cuda (int num_nodes, int start) {
    int *d_output;
    cudaMalloc((void **) &d_output, num_nodes*sizeof(int));
    int num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_output<<<num_blocks, THREADS_PER_BLOCK>>>(d_output, num_nodes, start);
    return d_output;
}

bool* init_bool_cuda () {
    bool *d_bool;
    cudaMalloc((void **) &d_bool, sizeof(bool));
    return d_bool;
}

int* bfs_cuda (Edge *h_edge_list, int num_nodes, int num_edges, int start) {
    Edge *d_edge_list = init_edge_list_cuda(h_edge_list, num_edges);
    int *d_output = init_output_cuda(num_nodes, start);
    bool h_done = false;
    bool *d_done = init_bool_cuda();

    int num_blocks = (num_edges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int i = 0; i < num_nodes; i++) {
        init_bool<<<1, THREADS_PER_BLOCK>>>(d_done, true);
        bfs<<<num_blocks, THREADS_PER_BLOCK>>>(d_edge_list, d_output, d_done, num_nodes, num_edges, i);
        cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_done) {
            break;
        }
    }

    int bytes = num_nodes*sizeof(int);
    int *h_output = (int*) malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_edge_list);
    cudaFree(d_output);
    return h_output;
}

int* bfs_serial (Edge *edge_list, int num_nodes, int num_edges, int start) {
    unordered_map<int, vector<int>> adj_list;
    for (int i = 0; i < num_edges; i++) {
        adj_list[edge_list[i].u].push_back(edge_list[i].v);
    }

    int *output = (int*) malloc(num_nodes*sizeof(int));
    bool *visited = (bool*) malloc(num_nodes*sizeof(bool));
    for (int i = 0; i < num_nodes; i++) {
        output[i] = INT_MAX;
        visited[i] = false;
    }
    output[start] = 0;
    visited[start] = true;

    vector<int> bfs_queue;
    bfs_queue.push_back(start);
    int iteration = 1;

    while (!bfs_queue.empty()) {
        vector<int> next_queue;
        while (!bfs_queue.empty()) {
            int node = bfs_queue.back();
            bfs_queue.pop_back();
            for (int neighbor: adj_list[node]) {
                if (!visited[neighbor]) {
                    output[neighbor] = iteration;
                    next_queue.push_back(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
        bfs_queue = next_queue;
        iteration++;
    }

    return output;
}

int main() {
    int num_nodes;
    int num_edges;
    cout << "Parsing Input File" << endl;
    Edge *edge_list = parse_file("graphs/dir_8.txt", num_nodes, num_edges);
    Edge *dummy = parse_file("graphs/dir_8.txt", num_nodes, num_edges);
    cout << "Parsed Input File" << endl << endl;
    
	{
		bfs_cuda(dummy, num_nodes, num_edges, 0);
		cudaThreadSynchronize();
	}
    cout << "Starting Parallel" << endl;
	auto start = chrono::steady_clock::now();
    int *output_p = bfs_cuda(edge_list, num_nodes, num_edges, 0);
	cudaThreadSynchronize();
	auto end = chrono::steady_clock::now();
	cout << chrono::duration_cast<chrono::microseconds>(end - start).count() << ", us" << endl;
    cout << "Finished Parallel" << endl << endl;

    cout << "Starting Serial" << endl;
    int *output_s = bfs_serial(edge_list, num_nodes, num_edges, 0);
    cout << "Finished Serial" << endl << endl;
    
    cout << "Checking Result" << endl;
    for (int i = 0; i < num_nodes; i++) {
        if (output_p[i] != output_s[i]) {
            cout << "FAILURE!!!" << endl;
            return 0;
        }
    }
    cout << "SUCCESS!!!" << endl;
    return 0;
}
