import heapq

class Node:
    def __init__(self, name, data, heuristic_cost=0):
        self.name = name
        self.data = data
        self.heuristic_cost = heuristic_cost
        self.neighbors = []

    def add_neighbor(self, neighbor, cost):
        self.neighbors.append((neighbor, cost))

class Edge:
    def __init__(self, start, end, cost):
        self.start = start
        self.end = end
        self.cost = cost

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, name, heuristic_cost=0):
        node = Node(name, heuristic_cost)
        self.nodes.append(node)
        return node

    def add_edge(self, start, end, cost):
        edge = Edge(start, end, cost)
        start.add_neighbor(end, cost)
        self.edges.append(edge)
        return edge

def astar(graph, start, goal):
    open_set = []
    closed_set = set()

    heapq.heappush(open_set, (start.heuristic_cost, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            path = reconstruct_path(came_from, goal)
            return path

        closed_set.add(current_node)

        for neighbor, cost in current_node.neighbors:
            if neighbor in closed_set:
                continue

            new_cost = cost_so_far[current_node] + cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + neighbor.heuristic_cost
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current_node

    return None

def reconstruct_path(came_from, goal):
    current = goal
    path = [current]

    while current in came_from and came_from[current] is not None:
        current = came_from[current]
        path.append(current)

    return path[::-1]


# Example usage:
if __name__ == "__main__":
    # Create a simple graph
    my_graph = Graph()

    node_a = my_graph.add_node("A", heuristic_cost=5)
    node_b = my_graph.add_node("B", heuristic_cost=3)
    node_c = my_graph.add_node("C", heuristic_cost=2)
    node_d = my_graph.add_node("D", heuristic_cost=1)
    node_e = my_graph.add_node("E", heuristic_cost=4)
    node_f = my_graph.add_node("F", heuristic_cost=2)

    my_graph.add_edge(node_a, node_b, cost=1)
    my_graph.add_edge(node_a, node_c, cost=3)
    my_graph.add_edge(node_b, node_d, cost=2)
    my_graph.add_edge(node_c, node_d, cost=1)
    my_graph.add_edge(node_c, node_e, cost=4)
    my_graph.add_edge(node_d, node_f, cost=3)
    my_graph.add_edge(node_e, node_f, cost=2)

    start_node = node_a
    goal_node = node_d

    # Find the shortest path using A*
    shortest_path = astar(my_graph, start_node, goal_node)

    if shortest_path:
        print(f"Shortest path from {start_node.name} to {goal_node.name}:")
        for node in shortest_path:
            print(node.name)
    else:
        print("No path found.")
