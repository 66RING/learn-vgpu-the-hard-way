#!/usr/bin/env python3
import sys
import random

if len(sys.argv) < 2:
	print("Missing command line argument!")
	exit(1)
num_nodes = 1 << int(sys.argv[1])
edge_prob = 0.5
max_weight = num_nodes
out_fname = '../graphs/dir_' + sys.argv[1] + '.txt'

matrix = []

for _ in range(num_nodes):
    l = []
    for _ in range(num_nodes):
        l.append(None)
    matrix.append(l)

num_edges = 0
for row in range(num_nodes):
	for col in range(num_nodes):
		if row != col and random.random() <= edge_prob:
			matrix[row][col] = random.randrange(1, num_nodes+1)
			num_edges += 1

out_file = open(out_fname, 'w')
out_file.write(str(num_nodes) + ' ' + str(num_edges))
for row in range(num_nodes):
	for col in range(num_nodes):
		if matrix[row][col]:
			out_file.write('\n' + str(row) + ' ' + str(col) + ' ' + str(matrix[row][col]))
out_file.close()
