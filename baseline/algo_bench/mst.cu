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
constexpr int THREADS_PER_BLOCK_2D = 32;


typedef struct {
    int u;
    int v;
    int w;
} Edge;

typedef struct {
    int cost;
    int index;
} MatrixEntry;

typedef struct {
    MatrixEntry mentry;
    int parent;
    bool valid;
} TreeNode;

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

__global__ void init_matrix (MatrixEntry *matrix, int num_nodes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_nodes && col < num_nodes) {
        int index = row * num_nodes + col;
        matrix[index].cost = (row == col) ? 0 : INT_MAX;
        matrix[index].index = index;
    }
}

__global__ void fill_matrix (Edge *edge_list, MatrixEntry *matrix, int num_nodes, int num_edges) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_edges) {
        int midx = edge_list[index].u * num_nodes + edge_list[index].v;
        matrix[midx].cost = edge_list[index].w;
    }
}

__global__ void init_tree (TreeNode *tree, int num_nodes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_nodes) {
        tree[index].mentry.cost = INT_MAX;
        tree[index].mentry.index = INT_MAX;
        tree[index].parent = INT_MAX;
        tree[index].valid = true;
    }
}

__global__ void init_output (Edge *output, int num_nodes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_nodes) {
        output[index].u = INT_MAX;
        output[index].v = INT_MAX;
        output[index].w = INT_MAX;
    }
}

__global__ void minimum_weight_edge (MatrixEntry *matrix, TreeNode *tree, int num_nodes) {
    __shared__ TreeNode sdata[THREADS_PER_BLOCK];
    int source = blockIdx.x;
    if (source >= num_nodes || !tree[source].valid) {
        return;
    }

    MatrixEntry *neighbors = &matrix[source * num_nodes];
    int tid = threadIdx.x;
    MatrixEntry min_entry;
    min_entry.cost = INT_MAX;
    min_entry.index = INT_MAX;
    int min_parent = INT_MAX;

    for (int i = tid; i < num_nodes; i += blockDim.x) {
        if (i == source || !tree[i].valid) {
            continue;
        }
        if (neighbors[i].cost < min_entry.cost) {
            min_entry = neighbors[i];
            min_parent = i;
        }
    }
    sdata[tid].mentry = min_entry;
    sdata[tid].parent = min_parent;
    sdata[tid].valid = (min_parent != INT_MAX);
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride &&
                sdata[tid + stride].mentry.cost < sdata[tid].mentry.cost) {
            sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        tree[source] = sdata[0];
    }
}

__global__ void rooted_tree (TreeNode *tree, int num_nodes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_nodes || !tree[index].valid) {
        return;
    }

    int parent = tree[index].parent;
    if (tree[parent].parent == index &&
            index < parent) {
        atomicExch(&tree[index].parent, index);
    }
}

__global__ void rooted_star (TreeNode *tree, int num_nodes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_nodes || !tree[index].valid) {
        return;
    }

    while (true) {
        int parent = tree[index].parent;
        if (tree[parent].parent == parent) {
            break;
        }

        atomicExch(&tree[index].parent, tree[parent].parent);
    }
}

__global__ void merge_fragments_cost_row (MatrixEntry *matrix, TreeNode *tree, int num_nodes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= num_nodes || col >= num_nodes ||
            !tree[row].valid || !tree[col].valid) {
        return;
    }

    int pcol = tree[col].parent;
    if (row != pcol && col != pcol) {
        int index = row * num_nodes + col;
        int parent = row * num_nodes + pcol;
        atomicMin(&matrix[parent].cost, matrix[index].cost);
    }
}

__global__ void merge_fragments_cost_col (MatrixEntry *matrix, TreeNode *tree, int num_nodes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= num_nodes || col >= num_nodes ||
            !tree[row].valid || !tree[col].valid) {
        return;
    }

    int prow = tree[row].parent;
    int pcol = tree[col].parent;
    if (row != prow && col == pcol) {
        int index = row * num_nodes + col;
        int parent = prow * num_nodes + col;
        atomicMin(&matrix[parent].cost, matrix[index].cost);
    }
}

__global__ void merge_fragments_index_row (MatrixEntry *matrix, TreeNode *tree, int num_nodes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= num_nodes || col >= num_nodes ||
            !tree[row].valid || !tree[col].valid) {
        return;
    }

    int pcol = tree[col].parent;
    if (row != pcol && col != pcol) {
        int index = row * num_nodes + col;
        int parent = row * num_nodes + pcol;
        if (matrix[index].cost == matrix[parent].cost) {
            atomicExch(&matrix[parent].index, matrix[index].index);
        }
    }
}

__global__ void merge_fragments_index_col (MatrixEntry *matrix, TreeNode *tree, int num_nodes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= num_nodes || col >= num_nodes ||
            !tree[row].valid || !tree[col].valid) {
        return;
    }

    int prow = tree[row].parent;
    int pcol = tree[col].parent;
    if (row != prow && col == pcol) {
        int index = row * num_nodes + col;
        int parent = prow * num_nodes + col;
        if (matrix[index].cost == matrix[parent].cost) {
            atomicExch(&matrix[parent].index, matrix[index].index);
        }
    }
}

__global__ void update_output (TreeNode *tree, Edge *output, int num_nodes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_nodes && tree[index].valid &&
            (index != tree[index].parent)) {
        output[index].u = tree[index].mentry.index / num_nodes;
        output[index].v = tree[index].mentry.index % num_nodes;
        output[index].w = tree[index].mentry.cost;
        tree[index].valid = false;
    }
}

MatrixEntry* init_matrix_cuda (Edge *h_edge_list, int num_nodes, int num_edges) {
    int in_bytes = num_edges*sizeof(Edge);
    int out_bytes = num_nodes*num_nodes*sizeof(MatrixEntry);
    Edge *d_edge_list;
    MatrixEntry *d_matrix;
    cudaMalloc((void **) &d_edge_list, in_bytes);
    cudaMalloc((void **) &d_matrix, out_bytes);
    cudaMemcpy(d_edge_list, h_edge_list, in_bytes, cudaMemcpyHostToDevice);
    

    dim3 dimBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    int num_blocks_2d = (num_nodes + THREADS_PER_BLOCK_2D - 1) /
            (THREADS_PER_BLOCK_2D);
    dim3 dimGrid(num_blocks_2d, num_blocks_2d);
    init_matrix<<<dimGrid, dimBlock>>>(d_matrix, num_nodes);
    int num_blocks = (num_edges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_matrix<<<num_blocks, THREADS_PER_BLOCK>>>(d_edge_list, d_matrix, num_nodes, num_edges);

    cudaFree(d_edge_list);
    return d_matrix;
}

TreeNode* init_tree_cuda (int num_nodes) {
    TreeNode *d_tree;
    cudaMalloc((void **) &d_tree, num_nodes*sizeof(TreeNode));
    int num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_tree<<<num_blocks, THREADS_PER_BLOCK>>>(d_tree, num_nodes);
    return d_tree;
}

Edge* init_output_cuda (int num_nodes) {
    Edge *d_output;
    cudaMalloc((void **) &d_output, num_nodes*sizeof(Edge));
    int num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_output<<<num_blocks, THREADS_PER_BLOCK>>>(d_output, num_nodes);
    return d_output;
}

int* init_int_cuda () {
    int* d_var;
    cudaMalloc((void **) &d_var, sizeof(int));
    return d_var;
}

void minimum_weight_edge_cuda (MatrixEntry *d_matrix, TreeNode *d_tree, int num_nodes) {
    minimum_weight_edge<<<num_nodes, THREADS_PER_BLOCK>>> (d_matrix, d_tree, num_nodes);
}

void rooted_tree_cuda (TreeNode *d_tree, int num_nodes) {
    int num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rooted_tree<<<num_blocks, THREADS_PER_BLOCK>>> (d_tree, num_nodes);
    rooted_star<<<num_blocks, THREADS_PER_BLOCK>>> (d_tree, num_nodes);
}

void merge_fragments_cuda (MatrixEntry *d_matrix, TreeNode *d_tree, int num_nodes) {
    dim3 dimBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    int num_blocks_2d = (num_nodes + THREADS_PER_BLOCK_2D - 1) /
            (THREADS_PER_BLOCK_2D);
    dim3 dimGrid(num_blocks_2d, num_blocks_2d);
    merge_fragments_cost_row<<<dimGrid, dimBlock>>>(d_matrix, d_tree, num_nodes);
    merge_fragments_cost_col<<<dimGrid, dimBlock>>>(d_matrix, d_tree, num_nodes);
    merge_fragments_index_row<<<dimGrid, dimBlock>>>(d_matrix, d_tree, num_nodes);
    merge_fragments_index_col<<<dimGrid, dimBlock>>>(d_matrix, d_tree, num_nodes);
}

void update_output_cuda (TreeNode *d_tree, Edge *d_output, int num_nodes) {
    int num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    update_output<<<num_blocks, THREADS_PER_BLOCK>>> (d_tree, d_output, num_nodes);
}

Edge* mst_cuda (Edge *h_edge_list, int num_nodes, int num_edges) {
    MatrixEntry *d_matrix = init_matrix_cuda(h_edge_list, num_nodes, num_edges);
    TreeNode *d_tree = init_tree_cuda(num_nodes);
    Edge *d_output = init_output_cuda(num_nodes);

    for (int i = num_nodes; i > 0; i >>= 1) {
        minimum_weight_edge<<<num_nodes, THREADS_PER_BLOCK>>> (d_matrix, d_tree, num_nodes);
        rooted_tree_cuda(d_tree, num_nodes);
        merge_fragments_cuda(d_matrix, d_tree, num_nodes);
        update_output_cuda(d_tree, d_output, num_nodes);
    }

    int bytes = num_nodes*sizeof(Edge);
    Edge *h_output = (Edge*) malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_tree);
    cudaFree(d_output);
    return h_output;
}

Edge* mst_serial (Edge *edge_list, int num_nodes, int num_edges) {
    unordered_map<int, vector<Edge>> adj_list;
    for (int i = 0; i < num_edges; i++) {
        adj_list[edge_list[i].u].push_back(edge_list[i]);
    }

    unordered_set<int> visited;
    auto compare = [](Edge a, Edge b) { return a.w > b.w; };
    priority_queue<Edge, vector<Edge>, decltype(compare)> mst_queue(compare);
    Edge* output = (Edge*) malloc((num_edges-1)*sizeof(Edge));
    for (int i = 0; i < adj_list[0].size(); i++) {
        mst_queue.push(adj_list[0][i]);
    }
    visited.insert(0);

    while (!mst_queue.empty() && visited.size() < num_nodes) {
        Edge next = mst_queue.top();
        mst_queue.pop();
        if (visited.count(next.v)) {
            continue;
        }
        output[visited.size()-1] = next;
        visited.insert(next.v);

        for (int i = 0; i < adj_list[next.v].size(); i++) {
            if (!visited.count(adj_list[next.v][i].v)) {
                mst_queue.push(adj_list[next.v][i]);
            }
        }
    }

    return output;
}

int main() {
    int num_nodes;
    int num_edges;
    cout << "Parsing Input File" << endl;
    Edge *edge_list = parse_file("graphs/undir_8.txt", num_nodes, num_edges);
    Edge *dummy = parse_file("graphs/undir_8.txt", num_nodes, num_edges);
    cout << "Parsed Input File" << endl << endl;
    
	{
		mst_cuda(dummy, num_nodes, num_edges);
		cudaThreadSynchronize();
	}
    cout << "Starting Parallel" << endl;
	auto start = chrono::steady_clock::now();
    Edge *mst_edges = mst_cuda(edge_list, num_nodes, num_edges);
	cudaThreadSynchronize();
	auto end = chrono::steady_clock::now();
	cout << chrono::duration_cast<chrono::microseconds>(end - start).count() << ", us" << endl;
    cout << "Finished Parallel" << endl ;
    int parallel_weight = 0;
    for (int i = 0; i < num_nodes; i++) {
        if (mst_edges[i].v != INT_MAX) {
           parallel_weight += mst_edges[i].w;
        }
    }
    cout << "Parallel MST Weight: " << parallel_weight << endl << endl;

    cout << "Starting Serial" << endl;
    Edge *mst_edges_serial = mst_serial(edge_list, num_nodes, num_edges);
    int serial_weight = 0;
    for (int i = 0; i < num_nodes-1; i++) {
        serial_weight += mst_edges_serial[i].w;
    }
    cout << "Finished Serial" << endl;
    cout << "Serial MST Weight: " << serial_weight << endl << endl;

    
    cout << "Checking Result" << endl;
    if (parallel_weight == serial_weight) {
        cout << "SUCCESS!!!" << endl;
    } else {
        cout << "FAILURE!!!" << endl;
    }
    return 0;
}
