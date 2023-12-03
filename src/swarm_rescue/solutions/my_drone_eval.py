# /usr/bin/env python3
import random
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean
import heapq

d = 40
l = 20

class Node:
    def __init__(self, name, gps_coord, heuristic_cost=0):
        self.name = name
        self.gps_coord = gps_coord
        self.heuristic_cost = heuristic_cost
        self.neighbors = []

    def add_neighbor(self, neighbor, cost):
        self.neighbors.append((neighbor, cost))

    def __getitem__(self, *args):
        assert len(args) == 1 and args[0] in [0, 1]
        return self.gps_coord[args[0]]

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
    
    def get_node_by_coords(self, target_coords):
        for node in self.nodes:
            if node.gps_coord == target_coords:
                return node
        return None

def astar(graph, start, goal):
    open_set = []
    closed_set = set()
    print("start:", start)
    print("goal: ",goal)
    heapq.heappush(open_set, (start.heuristic_cost, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while open_set:
        _, current_node = heapq.heappop(open_set)

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
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4
    class RecursiveGrid:
        cells: pd.DataFrame
        
        def __init__(self) -> None:
            self.cells = pd.DataFrame({"ulim": [], "llim": [], "dlim": [], "rlim": [], "waypoint_x": [], "waypoint_y": []})
            self.graph = Graph()

        def update(self, gps_coord, compass_angle, lidar):
            rays = np.roll(lidar[:-1], 90 + compass_angle)
            lims = [0]*4    # limit in the 4 directions
            
            # vertical_limits = self.cells[(self.cells["llim"] < gps_coord[0]) & (gps_coord[0] < self.cells["rlim"])] # We assume being in an unknown cell
            up = rays[0:90]
            up_wall = up * np.sin(np.pi*np.arange(90)/180)
            lims[0] = gps_coord[1] + 0.75*up_wall.max()
            
            down = rays[90:]
            down_wall = down * np.sin(np.pi*np.arange(90)/180)
            lims[2] = gps_coord[1] - 0.75*down_wall.max()
            
            # horizontal_limits = self.cells[(self.cells["dlim"] < gps_coord[1]) & (gps_coord[1] < self.cells["ulim"])]
            left = rays[45:135]
            left_wall = left * np.sin(np.pi*np.arange(90)/180)
            lims[1] = gps_coord[0] - 0.75*left_wall.max()
            

            right = np.concatenate((rays[-45:], rays[:45]))
            right_wall = right * np.sin(np.pi*np.arange(90)/180)
            lims[3] = gps_coord[0] + 0.75*right_wall.max()

            # Check the intersections with the other cells
            
            intercepting_mask = ((self.cells["llim"] > lims[3]) & (self.cells["dlim"] > lims[0]))  | \
                    ((self.cells["llim"] > lims[3]) & (self.cells["ulim"] < lims[2]))  | \
                    ((self.cells["rlim"] < lims[1]) & (self.cells["dlim"] > lims[0]))  | \
                    ((self.cells["rlim"] < lims[1]) & (self.cells["dlim"] < lims[2]))
            intercepting_cells = self.cells[~intercepting_mask]
            lims[0] = min(lims[0], intercepting_cells.loc[intercepting_cells["dlim"] > gps_coord[1], "dlim"].min())
            lims[1] = max(lims[1], intercepting_cells.loc[intercepting_cells["rlim"] < gps_coord[0], "rlim"].max())
            lims[2] = max(lims[2], intercepting_cells.loc[intercepting_cells["ulim"] < gps_coord[1], "ulim"].max())
            lims[3] = min(lims[3], intercepting_cells.loc[intercepting_cells["llim"] > gps_coord[0], "llim"].min())

            lims += [*gps_coord]
            self.cells.loc[-1] = lims
            self.cells.index += 1
             # Update the graph
            self.update_graph()
            
            # import matplotlib.pyplot as plt
            # import matplotlib.patches as patches

            # fig, ax = plt.subplots(1)

            # # Set the window size
            # ax.set_xlim([-450, 450])
            # ax.set_ylim([-350, 350])

            # # Loop over the rows of the dataframe and add each rectangle to the plot
            # for _, row in self.cells.iterrows():
            #     rect = patches.Rectangle((row['llim'], row['dlim']), row['rlim']-row['llim'], row['ulim']-row['dlim'], linewidth=1, edgecolor='r', facecolor='blue', fill=True)
            #     ax.add_patch(rect)
            # # Plot the graph nodes
            # for node in self.graph.nodes:
            #     print(node.gps_coord[0])
            #     x, y = node.gps_coord
            #     ax.scatter(x, y, color='g', s=50)  # Adjust the size (s) as needed
    
            # # Plot the graph edges manually
            # for edge in self.graph.edges:
            #     node1 = edge.start
            #     node2 = edge.end
            #     x1, y1 = node1.gps_coord
            #     x2, y2 = node2.gps_coord
            #     ax.plot([x1, x2], [y1, y2], color='g', linewidth=2)
            #     print(x1, x2)
            # plt.show()
            # _ = 1

        def update_graph(self):
            # Get the index and waypoints from the updated map
            index = self.cells.index[-1]
            waypoint_x = self.cells.iloc[-1]['waypoint_x']
            waypoint_y = self.cells.iloc[-1]['waypoint_y']
            #print(waypoint_x, waypoint_y)
            # Add the waypoint as a node in the graph
            waypoint_node = Node(name=index, gps_coord=(waypoint_x, waypoint_y))
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

        def __contains__(self, point):
            return np.any((self.cells['llim'] < point[0]) & (point[0] <= self.cells['rlim']) & 
                        (self.cells['dlim'] < point[1]) & (point[1] <= self.cells['ulim']))
        
        def get_cell_for_point(self, point):

    
            """
            Get the cell corresponding to the given point.

            Parameters:
            - point (tuple): Tuple containing the x and y coordinates of the point.

            Returns:
            - pd.Series or None: The cell data if the point is in a cell, else None.
            """
            in_cell = (
                (self.cells['llim'] <= point[0]) &
                (point[0] <= self.cells['rlim']) &
                (self.cells['dlim'] <= point[1]) &
                (point[1] <= self.cells['ulim'])
            )
            
            # Check if any cell contains the point
            if in_cell.any():
                return True  # The point is in a cell
            else:
                return False  
            
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

        self.forward = 0
        self.lateral = 0
        self.rotation = 0
        self.grasper = 0
        self.last_ts = 0

        # Boolean variables for state machine logic
        self.found_wounded = False
        self.found_rescue_center = False

        # The state is initialized to searching wounded person
        self.grid = MyDroneEval.RecursiveGrid()
        self.state = MyDroneEval.Activity.SEARCHING_WOUNDED
        self.following = MyDroneEval.ReachWrapper((np.inf, np.inf))

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        
        # Set the attributes needed later
        self.semantic_val = self.semantic_values()
        self.gps_val = self.measured_gps_position()
        self.compass_val = self.measured_compass_angle()
        self.lidar_val = savitzky_golay(self.lidar_values(), 21, 3)
        
        # Check if there is a visible body or the rescue center
        
        self.track_goals()

        #update the grid
        
        self.update_grid()
        
        # found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()
        # Compute its position and set the following attribute
        
        ####################################
        # TRANSITIONS OF THE STATE MACHINE #
        ####################################

        if self.state is self.Activity.SEARCHING_WOUNDED and self.found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
            self.found_wounded = False
            self.paths = astar(self.grid.graph, self.grid.last_visited_node, self.grid.rescue_point)
            self.path_index = 0
            print("paths", self.paths)

        # Should never happen
        elif self.state is self.Activity.GRASPING_WOUNDED and not self.found_wounded:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and self.found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.found_rescue_center = False
            self.state = self.Activity.SEARCHING_WOUNDED

        # Should never happen
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER


        #############################################################
        # COMMANDS FOR EACH STATE                                   #
        # Searching randomly, but when a wounded person is detected #
        # we use A* algorithm to backtrack to the rescue center     #
        #############################################################
        if self.state is self.Activity.SEARCHING_WOUNDED:
            self.search()

        elif self.state is self.Activity.GRASPING_WOUNDED or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            self.grasper = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            if self.following.distance(self.gps_val) < d:
                self.paths = self.paths[1:]
                self.following = MyDroneEval.ReachWrapper(self.paths[0])
            self.grasper = 1

        # Once the attribute self.following is set let the commands be decided by
        # our mechanical control function self.reach
        self.reach()

        return {"forward": self.forward,
                "lateral": self.lateral,
                "rotation": self.rotation,
                "grasper": self.grasper}
    
    def track_goals(self):
        bodies = [ray for ray in self.semantic_val if ray.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not ray.grasped]
        if len(bodies) > 0 and self.state not in [MyDroneEval.Activity.SEARCHING_RESCUE_CENTER, MyDroneEval.Activity.DROPPING_AT_RESCUE_CENTER]:
            positions = [(np.cos(body.angle + self.compass_val)*body.distance, np.sin(body.angle + self.compass_val)*body.distance) for body in bodies]
            x, y = zip(*positions)
            x, y = sum(x) / len(x), sum(y) / len(y)
            self.following = MyDroneEval.ReachWrapper((self.gps_val[0]+x, self.gps_val[1]+y))
            self.found_wounded = True
            self.last_ts = 0
        
        rescue = [ray for ray in self.semantic_val if ray.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER and not ray.grasped]
        if len(rescue) > 0:
            # positions = [(np.cos(body.angle + self.compass_val)*body.distance, np.sin(body.angle + self.compass_val)*body.distance) for body in bodies]
            # x, y = zip(*positions)
            # x, y = sum(x) / len(x), sum(y) / len(y)
            self.found_rescue_center = True
        else:
            self.found_rescue_center = False

    def update_grid(self):
        if self.gps_val not in self.grid:
            compass = convert_angle(self.compass_val) # Returns values in [0, 180]
            self.grid.update(self.gps_val, compass, self.lidar_val)
        else:
            compass = convert_angle(self.compass_val)

            # Check if the GPS is in a different cell than the last visited waypoint
            current_cell = self.grid.get_cell_for_point(self.gps_val)
            if getattr(self, 'last_visited_node', None) is not None:
                print(current_cell)
                last_visited_waypoint_cell = self.grid.get_cell_for_point(self.last_visited_node.gps_coord)
                print(last_visited_waypoint_cell)
                if last_visited_waypoint_cell is not None and (current_cell['waypoint_x'] != last_visited_waypoint_cell['waypoint_x'] or current_cell['waypoint_y'] != last_visited_waypoint_cell['waypoint_y']):
                    # Update the last visited waypoint
                    self.last_visited_node = self.grid.graph.get_node_by_coords(self.grid.get_waypoint_for_cell(current_cell))
                    print("last visited", self.last_visited_node)

    def search(self):
        """Logic for determining the next goal position in the map"""
        converted_compass = convert_angle(self.compass_val) # Returns values in [0, 180]
        goal_angle = convert_angle(np.arctan2(self.following.y - self.gps_val[1], self.following.x - self.gps_val[0]))
        d2goal = self.lidar_val[(90+goal_angle-converted_compass)%181]
        # print(self.following.distance(gps))
        # If the destination is reached or it is impossible to reach it from this position find a new one
        if not (d < self.following.distance(self.gps_val) < np.inf) or self.following.distance(self.gps_val) / d2goal > 1.02:
            # if gps not in self.grid:
            #     self.grid.update(gps, compass, self.lidar_val)
            # Check with the cells with neighboring ids and move to unseen scenarios
            rays = np.roll(self.lidar_val[:-1], 90+converted_compass)
            q80 = np.quantile(self.lidar_val, 0.8)
            guess = self.gps_val + (rays * np.vstack([np.cos(np.arange(0, 2*np.pi, 2*np.pi/180)),
                                                        np.sin(np.arange(0, 2*np.pi, 2*np.pi/180))])).T
            guess = guess[rays > q80]
            guess = np.apply_along_axis(lambda row: row if row not in self.grid else np.array([np.nan, np.nan]), 1, guess)
            guess = guess[(~np.isnan(guess)).any(axis=1)]

            if guess.size == 0:
                self.following = MyDroneEval.ReachWrapper(self.grid.cells.loc[1, ["waypoint_x", "waypoint_y"]])
                # print([self.following.x, self.following.y])
            else:
                self.following = MyDroneEval.ReachWrapper(guess[np.random.choice(guess.shape[0])])
                # print([self.following.x, self.following.y])
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

    # def process_semantic_sensor(self):
    #     """
    #     According to his state in the state machine, the Drone will move towards a wound person or the rescue center
    #     """
    #     command = {"forward": 0.5,
    #                "lateral": 0.0,
    #                "rotation": 0.0}
    #     angular_vel_controller_max = 1.0

    #     best_angle = 0

    #     found_wounded = False
    #     if (self.state is self.Activity.SEARCHING_WOUNDED
    #         or self.state is self.Activity.GRASPING_WOUNDED) \
    #             and self.semantic_val is not None:
    #         scores = []
    #         for data in self.semantic_val:
    #             # If the wounded person detected is held by nobody
    #             if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
    #                 found_wounded = True
    #                 v = (data.angle * data.angle) + \
    #                     (data.distance * data.distance / 10 ** 5)
    #                 scores.append((v, data.angle, data.distance))

    #         # Select the best one among wounded people detected
    #         best_score = 10000
    #         for score in scores:
    #             if score[0] < best_score:
    #                 best_score = score[0]
    #                 best_angle = score[1]

    #     found_rescue_center = False
    #     is_near = False
    #     angles_list = []
    #     if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
    #         or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
    #             and self.semantic_val:
    #         for data in self.semantic_val:
    #             if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
    #                 found_rescue_center = True
    #                 angles_list.append(data.angle)
    #                 is_near = (data.distance < 50)

    #         if found_rescue_center:
    #             best_angle = circular_mean(np.array(angles_list))

    #     if found_rescue_center or found_wounded:
    #         # simple P controller
    #         # The robot will turn until best_angle is 0
    #         kp = 2.0
    #         a = kp * best_angle
    #         a = min(a, 1.0)
    #         a = max(a, -1.0)
    #         command["rotation"] = a * angular_vel_controller_max

    #         # reduce speed if we need to turn a lot
    #         if abs(a) == 1:
    #             command["forward"] = 0.2

    #     if found_rescue_center and is_near:
    #         command["forward"] = 0
    #         command["rotation"] = random.uniform(0.5, 1)

    #     return found_wounded, found_rescue_center, command


    # def process_lidar_sensor(self, the_lidar_sensor):
    #     command = {"forward": 1.0,
    #                "lateral": 0.0,
    #                "rotation": 0.0}
    #     angular_vel_controller = 0.5

    #     values = the_lidar_sensor.get_sensor_values()

    #     if values is None:
    #         return command, False

    #     ray_angles = the_lidar_sensor.ray_angles
    #     size = the_lidar_sensor.resolution

    #     far_angle_raw = 0
    #     near_angle_raw = 0
    #     min_dist = 1000
    #     if size != 0:
    #         # far_angle_raw : angle with the longer distance
    #         # 这两个角度没看明白，是什么意思？
    #         far_angle_raw = ray_angles[np.argmax(values)]
    #         min_dist = min(values)
    #         # near_angle_raw : angle with the nearest distance
    #         near_angle_raw = ray_angles[np.argmin(values)]

    #     far_angle = far_angle_raw
    #     # If far_angle_raw is small then far_angle = 0
    #     if abs(far_angle) < 1 / 180 * np.pi:
    #         far_angle = 0.0

    #     near_angle = near_angle_raw
    #     #print(f"near angle: {near_angle}")
    #     far_angle = normalize_angle(far_angle)
    #     #print(f"far angle: {far_angle}")

    #     # The drone will turn toward the zone with the more space ahead
    #     #控制往哪个方向转，顺时针转还是逆时针转
    #     if size != 0:
    #         if far_angle > 0:
    #             command["rotation"] = angular_vel_controller
    #         elif far_angle == 0:
    #             command["rotation"] = 0
    #         else:
    #             command["rotation"] = -angular_vel_controller

    #     # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
    #     # size是什么
    #     collision = False
    #     if size != 0 and min_dist < 10:
    #         collision = True
    #         if near_angle > 0:
    #             command["rotation"] = -angular_vel_controller
    #         else:
    #             self.following = MyDroneEval.ReachWrapper(guess[np.random.choice(guess.shape[0])])
    #             print([self.following.x, self.following.y])
    #         self.last_ts = 0
        
    #     self.reach()