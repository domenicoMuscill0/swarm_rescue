import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import random
import pickle
import os

class Node:
    def __init__(self, name, coord):
        self.name = name
        self.coord = coord
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, node1, node2):
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)
        self.edges.append((node1, node2))

def build_3x3_grid():
    grid = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    return grid

def generate_random_nodes(grid, num_nodes):
    nodes = []
    available_coords = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]

    for _ in range(num_nodes):
        if not available_coords:
            break  # No more available coordinates
        random_coord = random.choice(available_coords)
        available_coords.remove(random_coord)
        nodes.append(Node(len(nodes), random_coord))

    return nodes

def generate_random_edges(nodes):
    edges = set()

    for node in nodes:
        num_neighbors = random.randint(1, min(3, len(nodes) - 1))  # Random number of neighbors
        random_neighbors = random.sample(nodes, num_neighbors)
        edges.update((node, neighbor) for neighbor in random_neighbors)

    return edges

def plot_graph(graph):
    for node in graph.nodes:
        plt.scatter(node.coord[0], node.coord[1], color='blue', s=100, label=f'Node {node.name}')
        for neighbor in node.neighbors:
            plt.plot([node.coord[0], neighbor.coord[0]], [node.coord[1], neighbor.coord[1]], color='gray')

    plt.title('Random Graphs on 3x3 Grid')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

def save_graph_to_dict(graph, filename):
    #Save nodes and edges of the graph in a dictionary using pickle
    graph_info = {'nodes': graph.nodes, 'edges':[(edge[0].name, edge[1].name) for edge in graph.edges]}
    #Use pickle to save the dictionary to a file
    with open(filename, 'wb') as file:
        pickle.dump(graph_info, file)

# Load the dictionary from the file using pickle
def load_graph(filename):
    with open(filename, 'rb') as file:
        graph_info = pickle.load(file)
    
    nodes = graph_info['nodes']
    edges = [(nodes[edge[0]], nodes[edge[1]]) for edge in graph_info['edges']]

    #Create a new graph
    loaded_graph = Graph()

    #Add nodes to tthe new graph
    for node in nodes:
        loaded_graph.add_node(node)
    for edge in edges:
        loaded_graph.add_edge(*edge)
    
    return loaded_graph
        
# Check for common nodes and merge graphs
def merge_graphs(graph1_filename, graph2_filename, merged_graph_filename):
    # Load graph1 from file
    graph1 = load_graph(graph1_filename)

    # Load graph2 from file
    graph2 = load_graph(graph2_filename)

    common_nodes = set(node.coord for node in graph1.nodes).intersection(node.coord for node in graph2.nodes)
    print(f"common node = {common_nodes}")

    if common_nodes:
        # If there are common nodes, merge the graph using the intersection
        merged_graph = Graph()

        # Add nodes and edges from graph1
        for node in graph1.nodes:
            merged_graph.add_node(node)
        for edge in graph1.edges:
            merged_graph.add_edge(*edge)

        # Add nodes and edges from graph2, excluding common nodes
        for node in graph2.nodes:
            if node.coord not in common_nodes:
                merged_graph.add_node(node)
        for edge in graph2.edges:
            if edge[0].coord not in common_nodes and edge[1].coord not in common_nodes:
                merged_graph.add_edge(*edge)    
        
        # Save the merged graph to file
                save_graph_to_dict(merged_graph, merged_graph_filename)
        
        return merged_graph, True
    else:
        return graph1, False


    
'''
# In the following main function, you will first generate a graph,
# Then save to the two files.

if __name__ == "__main__":
    # Build 3x3 grid
    grid = build_3x3_grid()

    # Generate random nodes and edges for two graphs
    nodes_graph1 = generate_random_nodes(grid, num_nodes=4)
    nodes_graph2 = generate_random_nodes(grid, num_nodes=4)

    edges_graph1 = generate_random_edges(nodes_graph1)
    edges_graph2 = generate_random_edges(nodes_graph2)

    # Create graphs
    graph1 = Graph()
    graph2 = Graph()

    # Add nodes and edges to graphs
    for node in nodes_graph1:
        graph1.add_node(node)
    for edge in edges_graph1:
        graph1.add_edge(*edge)

    for node in nodes_graph2:
        graph2.add_node(node)
    for edge in edges_graph2:
        graph2.add_edge(*edge)

    # Plot the graphs
    plot_graph(graph1)
    plot_graph(graph2)

    # Save graph1 and graph2 information to files
    save_graph_to_dict(graph1, 'graph1_info.pickle')
    save_graph_to_dict(graph2, 'graph2_info.pickle')


    # Merge graphs (if there are common nodes)
    merged_result = merge_graphs('graph1_info.pickle', 'graph2_info.pickle', 'merged_graph_info.pickle')
    merged_graph, merging_occurred = merged_result
    print(merging_occurred)

    # Plot the merged graph if merging occurred
    if merging_occurred:
        print("Hi!")
        plot_graph(merged_graph)


'''



if __name__ == "__main__":
    # Build 3x3 grid
    grid = build_3x3_grid()

    # Create graphs
    graph1 = Graph()
    graph2 = Graph()

    # Save graph1 and graph2 information to files
    graph1 = load_graph('graph1_info.pickle')
    graph2 = load_graph('graph2_info.pickle')

    # Plot the graphs
    plot_graph(graph1)
    plot_graph(graph2)

    # Merge graphs (if there are common nodes)
    merged_result = merge_graphs('graph1_info.pickle', 'graph2_info.pickle', 'merged_graph_info.pickle')
    merged_graph, merging_occurred = merged_result

    # Plot the merged graph if merging occurred
    if merging_occurred:
        plot_graph(merged_graph)
