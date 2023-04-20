#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>

#include <limits.h>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
using namespace std;

constexpr int THREADS_PER_BLOCK = 1024;
constexpr int THREADS_PER_BLOCK_2D = 32;


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

__global__ void init_matrix (int *matrix, int num_nodes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_nodes && col < num_nodes) {
        int index = row * num_nodes + col;
        matrix[index] = (row == col) ? 1 : 0;
    }
}

__global__ void fill_matrix (Edge *edge_list, int *matrix, int num_nodes, int num_edges) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_edges) {
        int midx = edge_list[index].u * num_nodes + edge_list[index].v;
        matrix[midx] = 1;
    }
}

__global__ void init_output (int *output, int num_nodes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_nodes) {
        output[index] = INT_MAX;
    }
}

__global__ void reachability (int *matrix, int num_nodes) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    int index = row * num_nodes + col;
    if (row >= num_nodes || col >= num_nodes ||
            matrix[index] == 1) {
        return;
    }

    int tid = threadIdx.x;
    int inter_node = blockIdx.z * blockDim.x + tid;
    if (inter_node >= num_nodes) {
        return;
    }

    int backward = row * num_nodes + inter_node;
    int forward = inter_node * num_nodes + col;
    if (matrix[backward] == 1 && matrix[forward] == 1) {
        matrix[index] = 1;
    }
}

__global__ void assign_scc (int *matrix, int *output, int num_nodes) {
    __shared__ int sdata[THREADS_PER_BLOCK];
    int node = blockIdx.x;
    if (node >= num_nodes) {
        return;
    }

    int tid = threadIdx.x;
    sdata[tid] = INT_MAX;
    for (int i = tid; i < num_nodes; i += blockDim.x) {
        if (matrix[node * num_nodes + i] == 1 &&
                matrix[i * num_nodes + node] == 1) {
            sdata[tid] = i;
            break;
        }
    }
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = min(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[node] = sdata[0];
    }
}

int* list_to_matrix_cuda (Edge *h_edge_list, int num_nodes, int num_edges) {
    int in_bytes = num_edges*sizeof(Edge);
    int out_bytes = num_nodes*num_nodes*sizeof(int);
    Edge *d_edge_list;
    int *d_matrix;
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

int* init_output_cuda (int num_nodes) {
    int *d_output;
    cudaMalloc((void **) &d_output, num_nodes*sizeof(int));
    int num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_output<<<num_blocks, THREADS_PER_BLOCK>>>(d_output, num_nodes);
    return d_output;
}

int* scc_cuda (Edge *h_edge_list, int num_nodes, int num_edges) {
    int *d_matrix = list_to_matrix_cuda(h_edge_list, num_nodes, num_edges);
    int* d_output = init_output_cuda(num_nodes);

    dim3 dimGrid(num_nodes, num_nodes,
            (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    for (int i = num_nodes; i > 0; i >>= 1) {
        reachability<<<dimGrid, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(int)>>>(d_matrix, num_nodes);
    }
    assign_scc<<<num_nodes, THREADS_PER_BLOCK>>>(d_matrix, d_output, num_nodes);

    int bytes = num_nodes*sizeof(int);
    int *h_output = (int*) malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_output);
    return h_output;
}

int* list_to_matrix_serial (Edge *edge_list, int num_nodes, int num_edges) {
    int *matrix = (int*) malloc(num_nodes*num_nodes*sizeof(int));
    for (int row = 0; row < num_nodes; row++) {
        for (int col = 0; col < num_nodes; col++) {
            matrix[row * num_nodes + col] = (row == col) ? 1 : 0;
        }
    }

    for (int i = 0; i < num_edges; i++) {
        Edge e = edge_list[i];
        matrix[e.u * num_nodes + e.v] = 1;
    }
    return matrix;
}

int* scc_serial (Edge *edge_list, int num_nodes, int num_edges) {
    int *matrix = list_to_matrix_serial(edge_list, num_nodes, num_edges);

    for (int k = 0; k < num_nodes; k++) {
        for (int row = 0; row < num_nodes; row++) {
            for (int col = 0; col < num_nodes; col++) {
                int index = row * num_nodes + col;
                if (matrix[index] == 1) {
                    continue;
                }
                int backward = row * num_nodes + k;
                int forward = k * num_nodes + col;
                if (matrix[backward] == 1 && matrix[forward] == 1) {
                    matrix[index] = 1;
                }
            }
        }
    }

    int *output = (int*) malloc(num_nodes*sizeof(int));
    for (int node = 0; node < num_nodes; node++) {
        for (int i = 0; i < num_nodes; i++) {
            if (matrix[node * num_nodes + i] == 1 &&
                    matrix[i * num_nodes + node] == 1) {
                output[node] = i;
                break;
            }
        }
    }
    return output;
}

int main() {
    int num_nodes;
    int num_edges;
    cout << "Parsing Input File" << endl;
    Edge *edge_list = parse_file("graphs/dir_8_sparse.txt", num_nodes, num_edges);
    Edge *dummy = parse_file("graphs/dir_8_sparse.txt", num_nodes, num_edges);
    cout << "Parsed Input File" << endl << endl;

	{
		scc_cuda(edge_list, num_nodes, num_edges);
		cudaThreadSynchronize();
	}
    cout << "Starting Parallel" << endl;
	auto start = chrono::steady_clock::now();
    int *output_p = scc_cuda(edge_list, num_nodes, num_edges);
	cudaThreadSynchronize();
	auto end = chrono::steady_clock::now();
	cout << chrono::duration_cast<chrono::microseconds>(end - start).count() << ", us" << endl;
    cout << "Finished Parallel" << endl << endl;

    cout << "Starting Serial" << endl;
    int *output_s = scc_serial(edge_list, num_nodes, num_edges);
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
