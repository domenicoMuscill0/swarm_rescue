# /usr/bin/env python3
import random
from enum import Enum
import time
from typing import Optional
import numpy as np
import pandas as pd

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
import heapq
import numba as nb


d = 40

def reconstruct_path(came_from, goal):
    current = goal
    path = [current]

    while current in came_from and came_from[current] is not None:
        current = came_from[current]
        path.append(current)

    return path[::-1]

def astar(start, goal):
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (start.heuristic_cost, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while open_set:
        _, current_node = heapq.heappop(open_set)

        if current_node == goal:
            path = reconstruct_path(came_from, goal)
            return path, cost_so_far[goal]  # So that we can compute optimal new restart

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

@nb.njit(fastmath=True,error_model="numpy")
def contained(cells, points):
    res = [False]*len(points)
    for j in range(len(points)):
        for i in range(len(cells)):
            if (cells[i, 1] < points[j, 0]) & (points[j, 0] <= cells[i, 3]) & \
                    (cells[i, 2] < points[j, 1]) & (points[j, 1] <= cells[i, 0]):
                res[j] = True
                break
        else:
            res[j] = False
    return np.array(res)

@nb.njit(fastmath=True, error_model="numpy")
def intercept_grid(cells, lims, gps_coord):
    for i in range(len(cells)):
        # If there is interception
        if  (((lims[2] <= cells[i, 2] <= lims[0]) | (lims[2] <= cells[i, 0] <= lims[0]))  & \
            ((lims[1] <= cells[i, 1] <= lims[3]) | (lims[1] <= cells[i, 3] <= lims[3])))  | \
            (((cells[i, 2] <= lims[2] <= cells[i, 0]) | (cells[i, 2] <= lims[0] <= cells[i, 0]))  & \
            ((cells[i, 1] <= lims[1] <= cells[i, 3]) | (cells[i, 1] <= lims[3] <= cells[i, 3]))):
            if   cells[i, 2] > gps_coord[1]:
                lims[0] = min(cells[i, 2], lims[0])
            elif cells[i, 3] < gps_coord[0]:
                lims[1] = max(cells[i, 3], lims[1])
            elif cells[i, 0] < gps_coord[1]:
                lims[2] = max(cells[i, 0], lims[2])
            elif cells[i, 1] > gps_coord[0]:
                lims[3] = min(cells[i, 1], lims[3])

class RecursiveGrid:
    cells: pd.DataFrame
    
    def __init__(self) -> None:
        self.cells = pd.DataFrame({"ulim": [], "llim": [], "dlim": [], "rlim": [], "waypoint_x": [], "waypoint_y": []})
        self.graph = Graph()

    def update(self, gps_coord, compass_angle, lidar):
        rays = np.roll(lidar[:-1], 90 + compass_angle)
        lims = np.zeros((6,))    # limit in the 4 directions
        
        up = rays[0+22:90-22]
        up_wall = up * np.sin(np.pi*np.arange(90-2*22)/180)
        lims[0] = gps_coord[1] + up_wall.max()
        
        down = rays[90+22:-22]
        down_wall = down * np.sin(np.pi*np.arange(90-2*22)/180)
        lims[2] = gps_coord[1] - down_wall.max()
        
        left = rays[45+22:135-22]
        left_wall = left * np.sin(np.pi*np.arange(90-2*22)/180)
        lims[1] = gps_coord[0] - left_wall.max()
        

        right = np.concatenate((rays[-45+22:], rays[:45-22]))
        right_wall = right * np.sin(np.pi*np.arange(90-2*22)/180)
        lims[3] = gps_coord[0] + right_wall.max()

        # Impose a minimum cell size if the number of cells becomes a problem
        # if((lims[0]-lims[2])*(lims[3] - lims[1]) < 30):
        #     return

        # Check the intersections with the other cells
        
        intercept_grid(self.cells.values, lims, gps_coord)


        lims[-2:] = [*gps_coord]
        self.__add__(lims)

        # Update the graph
        self.update_graph(rays, self.cells.values)

        # Clean old free spots that now are no longer used
        self.graph.clean_spots(gps_coord, self)
        
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches

        # fig, ax = plt.subplots(1)

        # ## Set the window size
        # ax.set_xlim([-400, 400])  # Map intermediate 01
        # ax.set_ylim([-250, 250])

        # # ax.set_xlim([-830, 830])  # Map medium 01
        # # ax.set_ylim([-561, 561])

        # # # Loop over the rows of the dataframe and add each rectangle to the plot
        # for _, row in self.cells.iterrows():
        #     rect = patches.Rectangle((row['llim'], row['dlim']), row['rlim']-row['llim'], row['ulim']-row['dlim'], linewidth=1, edgecolor='r', facecolor='blue', fill=True)
        #     ax.add_patch(rect)
        # # # Plot the graph nodes
        # for node in self.graph.nodes:
        #     x, y = node.gps_coord
        #     ax.scatter(x, y, color='g', s=50)  # Adjust the size (s) as needed

        #  # Plot the graph edges manually
        # for edge in self.graph.edges:
        #     node1 = edge.start
        #     node2 = edge.end
        #     x1, y1 = node1.gps_coord
        #     x2, y2 = node2.gps_coord
        #     ax.plot([x1, x2], [y1, y2], color='g', linewidth=2)
        # plt.show()
        # _ = 1

    def update_graph(self, rays, other_cells):
        # Get the index and waypoints from the updated map
        index = self.cells.index[-1]
        waypoint_x = self.cells.iloc[-1]['waypoint_x']
        waypoint_y = self.cells.iloc[-1]['waypoint_y']
        # Add the waypoint as a node in the graph
        waypoint_node = Node(name=index, gps_coord=(waypoint_x, waypoint_y))

        # Find undiscovered spots

        q80 = np.quantile(rays, 0.8)
        spots = np.array((waypoint_x, waypoint_y)) + (rays * np.vstack([np.cos(np.arange(0, 2*np.pi, 2*np.pi/180)),
                                                np.sin(np.arange(0, 2*np.pi, 2*np.pi/180))])).T
        spots = spots[rays > q80]
        spots = spots[~contained(self.cells.values, spots)]
        waypoint_node.add_spots(spots)

        self.graph.add_node(waypoint_node)

        # Connect the waypoint node to adjacent nodes based on limits (lims)
        #self.connect_adjacent_nodes(waypoint_node, gps_coord)
        # If there is a last visited node, add an edge to the new waypoint node
        if getattr(self, 'last_visited_node', None) is not None:
            distance = np.linalg.norm(np.array(waypoint_node.gps_coord) - np.array(self.last_visited_node.gps_coord))
            self.graph.add_edge(self.last_visited_node, waypoint_node, cost=distance)
        else:
            self.rescue_point = waypoint_node
        # Update the last visited node
        self.last_visited_node = waypoint_node

    def __add__(self, other):
        self.cells.loc[-1] = other
        self.cells.index += 1

    def __contains__(self, points):
        return np.any((self.cells["llim"] < points[0]) & (points[0] <= self.cells["rlim"]) & \
                (self.cells["dlim"] < points[1]) & (points[1] <= self.cells["ulim"]))
    
    def get_cell_for_point(self, point):
        """
        Get the cell corresponding to the given point.

        Parameters:
        - point (tuple): Tuple containing the x and y coordinates of the point.

        Returns:
        - pd.Series or None: The cell data if the point is in a cell, else None.
        - the index of the cell in the grid DataFrame. -1 if not found
        """
        in_cell_mask = (
            (self.cells['llim'] <= point[0]) &
            (point[0] <= self.cells['rlim']) &
            (self.cells['dlim'] <= point[1]) &
            (point[1] <= self.cells['ulim'])
        )
        
        if in_cell_mask.any():  # Check if any cell contains the point
            return self.cells[in_cell_mask].iloc[0]#, np.where(in_cell_mask)[0].item()  # Return the first matching cell
        else:
            return None# , -1
        
    def get_node_by_point(self, pos):
        waypoint = self.get_cell_for_point(pos)
        return self.graph.get_node_by_waypoint((waypoint["waypoint_x"], waypoint["waypoint_y"]))
    
    
    def restart(self, gps_coord):
        waypoint_node = self.get_node_by_point(gps_coord)
        return self.graph.choose_restart(waypoint_node)
        

class Node:
    def __init__(self, name, gps_coord, heuristic_cost=0):
        self.name = name
        self.gps_coord = gps_coord
        self.heuristic_cost = heuristic_cost
        self.neighbors = []
        self.spots = []

    def add_neighbor(self, neighbor, cost):
        self.neighbors.append((neighbor, cost))
    
    def add_spots(self, spots):
        self.spots = spots

    def dot(self, other):
        return self.gps_coord[0]*other[0] + self.gps_coord[1]*other[1]
    
    def cross(self, other):
        return self.gps_coord[0]*other[1] - self.gps_coord[1]*other[0]

    def norm(self):
        return np.sqrt(self.gps_coord[0]**2 + self.gps_coord[1]**2)

    def __sub__(self, other):
        return Node("MOCK NODE", (self.gps_coord[0] - other[0], self.gps_coord[1] - other[1]))
    
    def __add__(self, other):
        return Node("MOCK NODE", (self.gps_coord[0] + other[0], self.gps_coord[1] + other[1]))
    
    def __mul__(self, scalar):
        return Node("MOCK NODE", (self.gps_coord[0]*scalar, self.gps_coord[1]*scalar))

    def __getitem__(self, *args):
        assert len(args) == 1 and args[0] in [0, 1]
        return self.gps_coord[args[0]]

    def __repr__(self) -> str:
        return f"({self.gps_coord[0]:.2f}, {self.gps_coord[1]:.2f})"
    
    def __str__(self):
        return f"Node(name={self.name}, gps_coord={self.gps_coord}, heuristic_cost={self.heuristic_cost}, neighbors={self.neighbors})"

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
    
    def add_node(self, node):
        self.nodes.append(node)
        return node
    
    def add_edge(self, start, end, cost):
        edge = Edge(start, end, cost)
        start.add_neighbor(end, cost)
        end.add_neighbor(start, cost)
        self.edges.append(edge)
        return edge
    
    def get_node_by_waypoint(self, target_coords):
        for node in self.nodes:
            if node.gps_coord == target_coords:
                return node
        return None
    
    def get_nearest_node(self, gps):
        best_d = np.inf
        best_node = None
        for node in self.nodes:
            d = (node - gps).norm()
            if d < best_d:
                best_d = d
                best_node = node
        return best_node
    
    def clean_spots(self, new_pos, grid: RecursiveGrid):
        new_node = grid.get_node_by_point(new_pos)
        for n, _ in new_node.neighbors:
            n.spots = n.spots[~contained(grid.cells.values, n.spots)]

    def choose_restart(self, waypoint):
        min_cost = np.inf
        min_path = None
        # goal = min(self.nodes, key=lambda n: astar(waypoint, n)[1] / (1+len(n.spots)))
        for i in range(len(self.nodes)):
            n_spots = len(self.nodes[i].spots)
            if n_spots == 0:    continue

            path, cost = astar(waypoint, self.nodes[i])
            if cost == 0:   continue
            if min_cost > cost / n_spots:
                min_cost = cost / n_spots
                min_path = path
        # return astar(waypoint, goal)[0]
        return min_path


def convert_angle(angle):
    if angle >= 0:
        return int(np.round(90 * angle / np.pi))
    else:
        return int(np.round(180 + 90 * angle / np.pi))
 
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter. The Savitzky-Golay filter removes high frequency noise from data. It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
       the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def gomperz(x, a=0.008, b=20, rate=0.1) -> float:
    return 1 - a*np.exp(-np.exp(-rate*(x-b)))


class MyDroneEval(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        SEARCHING_RESCUE_CENTER = 2    
    class ReachWrapper:
        x: float = np.nan
        y: float = np.nan
        def __init__(self, obj) -> None:
            try:
                self.x = obj.x
                self.y = obj.y
            except AttributeError:
                self.x = obj[0]
                self.y = obj[1]
        def distance(self, other):
            try:
                x = other.x
                y = other.y
            except AttributeError:
                x = other[0]
                y = other[1]
            return np.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)
        random.seed(identifier)

        # Internal variables for the actuators
        self.forward = 0
        self.lateral = 0
        self.rotation = 0
        self.grasper = 0
        self.last_ts = 0

        # Margin to keep from obstacles
        self.margin = 30

        # Boolean variables for state machine logic
        self.found_wounded = False
        self.found_rescue_center = False

        # The state is initialized to searching wounded person
        self.grid = RecursiveGrid()
        self.state = MyDroneEval.Activity.SEARCHING_WOUNDED
        self.following = MyDroneEval.ReachWrapper((np.inf, np.inf))
        self.paths = []

        # Data to be sent through the communication medium
        self.msg_data = {"id": self.identifier,
                         "following": self.following,
                         "no-comm zones": [],   # Accumulate the cells of non-communication zones and send them outside of the region
                         "destruction zones": []}   # If a destruction zone is reached accumulate the cells in it. Mandatory to send to every new drone


    def define_message_for_all(self):
        """
        Define the communication system among drones
        """
        return self.msg_data
    
    def process_messages(self):
        # To add the logic for sending and receiving local grid cells for non-common drones
        messages = self.communicator.received_messages
        for msg in messages:
            if len(msg["destruction zones"] > 0): 
                for dzone in msg["destruction zones"]:
                    if dzone not in self.msg_data["destruction zones"]:
                        self.msg_data["destruction zones"].append(dzone)
            if len(msg["no-comm zones"] > 0):
                for nczone in msg["no-comm zones"]:
                    if nczone not in self.grid:
                        self.grid += nczone
            

    def control(self):
        start = time.time()
        
        # Set the attributes needed later
        self.lidar_val = savitzky_golay(self.lidar_values(), 21, 3) # If lidar_values are None it means that we are about to die in a destruction zone
        self.set_gps_and_compass_val()
        self.semantic_val = self.semantic_values()
        
        # Check if there is a visible body or the rescue center
        
        self.track_goals()

        #update the grid
        
        self.update_grid()
        
        # Compute its position and set the following attribute
        
        ####################################
        # TRANSITIONS OF THE STATE MACHINE #    0.05 max
        ####################################

        if self.state is self.Activity.SEARCHING_WOUNDED and self.found_wounded and self.following.distance(self.gps_val) < d:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
            self.found_wounded = False
            self.margin = 50    # Increase the margin to take into account the new bigger hitbox
            self.paths, _ = astar(self.grid.last_visited_node, self.grid.rescue_point)


        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and self.found_rescue_center and self.following.distance(self.gps_val) < d:
            self.found_rescue_center = False
            self.margin = 30
            self.state = self.Activity.SEARCHING_WOUNDED


        #############################################################
        # COMMANDS FOR EACH STATE                                   #
        # Searching randomly, but when a wounded person is detected #
        # we use A* algorithm to backtrack to the rescue center     #
        #############################################################
        self.clean_material_following() # Clean from the lidar the distances to points of interest
        if self.lidar_val.min() < self.margin:
            self.solve_collision()
        else:
            # If there is a path try to optimize it
            if len(self.paths) > 1:     self.optimize_path()
            # Reset the margin to its normal value if the collision is solved. Enables dynamic margin adaptation
            self.margin = min(50, self.margin+1)
            if self.state is self.Activity.SEARCHING_WOUNDED and not self.found_wounded:
                self.search()
                self.grasper = 0

            elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and not self.found_rescue_center:
                if self.following.distance(self.gps_val) < d:

                    self.following = MyDroneEval.ReachWrapper(self.paths[0])
                    if(len(self.paths)>1):
                        self.paths = self.paths[1:]
                self.grasper = 1

            # Once the attribute self.following is set let the commands be decided by
            # our mechanical control function self.reach
            self.reach()
            end = time.time()
            if (end - start)*10000 > 2000:
                print(f"Time spent by 'control': {(end-start)*10000} s")
            
        # print(self.margin)
        self.validate_commands()
        return {"forward": self.forward,
                "lateral": self.lateral,
                "rotation": self.rotation,
                "grasper": self.grasper}
    
    def set_gps_and_compass_val(self):
        gps_pos = self.measured_gps_position()
        compass = self.measured_compass_angle()

        # If we reach a No-GPS zone we compute the new position through the odometer
        if (gps_pos is None) or (compass is None):
            dist, alpha, theta = self.odometer_values()
            self.gps_val[0] += dist*np.cos(alpha + self.compass_val)
            self.gps_val[1] += dist*np.sin(alpha + self.compass_val)
            self.compass_val += theta
        else:
            self.gps_val = gps_pos
            self.compass_val = compass
    
    def track_goals(self):
        """Tracks the positions of nearby bodies or rescue centers so that they can be reached without error."""
        bodies = [ray for ray in self.semantic_val if ray.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not ray.grasped]
        if len(bodies) > 0 and self.state is MyDroneEval.Activity.SEARCHING_WOUNDED:
            bodies = min(bodies, key=lambda r: r.distance)
            x, y = np.cos(bodies.angle + self.compass_val)*bodies.distance, np.sin(bodies.angle + self.compass_val)*bodies.distance
            self.following = MyDroneEval.ReachWrapper((self.gps_val[0]+x, self.gps_val[1]+y))
            self.found_wounded = True
            self.last_ts = 0
        else: self.found_wounded = False
        
        rescue = [ray for ray in self.semantic_val if ray.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER]
        if len(rescue) > 0 and self.state is MyDroneEval.Activity.SEARCHING_RESCUE_CENTER:
            rescue = min(rescue, key=lambda r: r.distance)
            x, y = np.cos(rescue.angle + self.compass_val)*rescue.distance, np.sin(rescue.angle + self.compass_val)*rescue.distance
            self.following = MyDroneEval.ReachWrapper((self.gps_val[0]+x, self.gps_val[1]+y))
            self.found_rescue_center = True
        else: self.found_rescue_center = False

    def optimize_path(self):
        orientation = Node("DRONE ORIENTATION", (np.cos(self.compass_val), np.sin(self.compass_val)))
        O = Node("DRONE_POS", self.gps_val)
        for i in range(1, len(self.paths)-1):
            # Search the farthest point in the path visible from here and substitute the path before with that
            start_edge = self.paths[i] - self.gps_val
            end_edge = self.paths[i+1] - self.gps_val

            # Get the angle between the drone orientation and the start of the edge
            angle_start = np.arccos(start_edge.dot(orientation) / (start_edge.norm() * orientation.norm()))
            angle_start = convert_angle(angle_start)

            # Get the hemisphere of the edge end node with respect to the drone orientation
            hemisphere_start = 1 if orientation.cross(start_edge) > 0 else -1

            # Get the angle between the drone orientation and the end of the edge
            angle_end = np.arccos(end_edge.dot(orientation) / (end_edge.norm() * orientation.norm()))
            angle_end = convert_angle(angle_end)

            # Get the hemisphere of the edge end node with respect to the drone orientation
            hemisphere_end = 1 if orientation.cross(end_edge) > 0 else -1

            # We define the indices for the lidar sensor that explain from which rays we consider
            begin = min(90+hemisphere_start*angle_start, 90+hemisphere_end*angle_end)
            end = max(90+hemisphere_start*angle_start, 90+hemisphere_end*angle_end)

            # we allow only states in which the drone is at least directed towards the goal
            if abs(begin-90) > 22 or abs(end - 90) > 22:  break
            # we want to optimize only if the next node is in sight
            if start_edge.norm() > self.lidar_val[begin-1:begin+1].min():  break
            
            alpha2 = np.cos(2*np.pi/180 *np.arange(0, end-begin + 1))**2
            
            # If also the ending node is in sight we go to the next edge
            if end_edge.norm() < self.lidar_val[end-1:end+1].min():  continue

            P = self.paths[i] - O
            P2 = P.norm()**2
            Q = self.paths[i+1] - self.paths[i]
            Q2 = Q.norm()**2
            PQ = P.dot(Q)
            t = (-(P2*PQ*(1-alpha2**2)) - np.sqrt((P2**2 * PQ**2 * (1-alpha2)**2) - ((1-alpha2) * P2**2 * (PQ**2 - P2*Q2*alpha2)))) / (PQ**2 - P2*Q2*alpha2)
            line_distance = (P+Q*t).norm()
            s = t[np.argmin(np.abs(self.lidar_val[begin:end+1] - line_distance))]
            
            self.paths = [O + P + Q*s] + self.paths[i+1:]
            break

    def update_grid(self):
        """Performs the update of the grid adding a new cell if we have to discover another portion of the map
           or to update the value of the last visited node."""
        if self.gps_val not in self.grid:
            compass = convert_angle(self.compass_val) # Returns values in [0, 180]

            self.grid.last_visited_node = self.grid.graph.get_nearest_node(self.gps_val)
            self.grid.update(self.gps_val, compass, self.lidar_val)

    def clean_material_following(self):
        for s in self.semantic_val:
            if (s.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not s.grasped) or \
            (s.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER and self.state == MyDroneEval.Activity.SEARCHING_RESCUE_CENTER):# or \
            # (s.entity_type == DroneSemanticSensor.TypeEntity.DRONE and self.state == MyDroneEval.Activity.SEARCHING_RESCUE_CENTER):     # Come se fosse n'ambulanza
                entity_ray = (convert_angle(s.angle)+90)%180
                self.lidar_val[entity_ray-5:entity_ray+5] = np.inf

    def solve_collision(self):
        """If there is a collision start turning towards the most secure direction and then start accelerating"""

        # Decrease the collision margin in case the drone gets stuck for being too cautious
        self.margin = max(1, self.margin-1)
        obstacle_angle = np.argmin(self.lidar_val) - 90
        obstacle_distance = self.lidar_val[obstacle_angle + 90]
        self.forward = -np.cos(np.pi*(obstacle_angle / 90)) * (1 - abs(obstacle_distance * np.cos(np.pi*(obstacle_angle / 90))) / self.margin)
        self.lateral = -np.sin(np.pi*(obstacle_angle / 90)) * (1 - abs(obstacle_distance * np.sin(np.pi*(obstacle_angle / 90))) / self.margin)
        
        alpha = np.arctan2(self.following.y - self.gps_val[1], self.following.x - self.gps_val[0]) - self.compass_val

        rot = min(abs(alpha)*np.exp(abs(alpha) / (2*np.pi)), 1)
        self.rotation = np.sign(alpha)*rot*gomperz(self.last_ts)

    def search(self):
        """Logic for determining the next goal position in the map"""
        converted_compass = convert_angle(self.compass_val) # Returns values in [0, 180]
        goal_angle = convert_angle(np.arctan2(self.following.y - self.gps_val[1], self.following.x - self.gps_val[0]))
        d2goal = self.lidar_val[(90+goal_angle-converted_compass)%181]

        # If the destination is reached or it is impossible to reach it from this position find a new one
        if not (d < self.following.distance(self.gps_val) < np.inf) or self.following.distance(self.gps_val) / d2goal > 1.02:
            # Check with the cells with neighboring ids and move to unseen scenarios
            if len(self.paths) > 1:        # If we have already chosen a path to restart we follow it
                self.paths = self.paths[1:]
                self.following = MyDroneEval.ReachWrapper(self.paths[0])
            else:                          # otherwise if we reached a no-go-further point...
                waypoint = self.grid.get_cell_for_point(self.gps_val)
                current_node = self.grid.graph.get_node_by_waypoint((waypoint["waypoint_x"], waypoint["waypoint_y"]))
                next_spots = current_node.spots
                if next_spots.size > 0:    # Some new spots available from this position
                    next_destination = next_spots[np.random.choice(next_spots.shape[0])]
                    self.following = MyDroneEval.ReachWrapper(next_destination)
                else:                      # No new spots available from this position
                    self.paths = self.grid.restart(self.gps_val)
                    self.following = MyDroneEval.ReachWrapper(self.paths[0]) # PROBLEM
            self.last_ts = 0

    def reach(self):
        """Reaches the entity defined in self.following."""

        # Obtain the angle to turn from current orientation
        alpha = np.arctan2(self.following.y - self.gps_val[1], self.following.x - self.gps_val[0]) - self.compass_val

        rot = min(abs(alpha)*np.exp(abs(alpha) / (2*np.pi)), 1)
        self.rotation = np.sign(alpha)*rot*gomperz(self.last_ts)
        self.forward = 1 if abs(rot) < 0.1 else 0.40
        self.lateral = -np.sign(alpha)*rot**2
        self.last_ts += 1

    def validate_commands(self):
        self.forward = np.clip(self.forward, a_min=-1, a_max=1)
        self.lateral = np.clip(self.lateral, a_min=-1, a_max=1)
        self.rotation = np.clip(self.rotation, a_min=-1, a_max=1)