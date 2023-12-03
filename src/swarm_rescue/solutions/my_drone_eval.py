"""
This program can be launched directly.
Example of how to use semantic sensor, grasping and dropping
"""

import os
import sys
import random
import math
from typing import Optional, List, Type
from enum import Enum
import numpy as np
import pandas as pd
from spg.utils.definitions import CollisionTypes

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, circular_mean
import heapq


class Node:
    def __init__(self, name, gps_coord, heuristic_cost=0):
        self.name = name
        self.gps_coord = gps_coord
        self.heuristic_cost = heuristic_cost
        self.neighbors = []

    def add_neighbor(self, neighbor, cost):
        self.neighbors.append((neighbor, cost))

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

def gomperz(x, a=0.008, b=30, rate=0.1) -> float:
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
            
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, ax = plt.subplots(1)

            # Set the window size
            ax.set_xlim([-450, 450])
            ax.set_ylim([-350, 350])

            # Loop over the rows of the dataframe and add each rectangle to the plot
            for _, row in self.cells.iterrows():
                rect = patches.Rectangle((row['llim'], row['dlim']), row['rlim']-row['llim'], row['ulim']-row['dlim'], linewidth=1, edgecolor='r', facecolor='blue', fill=True)
                ax.add_patch(rect)
            # Plot the graph nodes
            for node in self.graph.nodes:
                print(node.gps_coord[0])
                x, y = node.gps_coord
                ax.scatter(x, y, color='g', s=50)  # Adjust the size (s) as needed
    
            # Plot the graph edges manually
            for edge in self.graph.edges:
                node1 = edge.start
                node2 = edge.end
                x1, y1 = node1.gps_coord
                x2, y2 = node2.gps_coord
                ax.plot([x1, x2], [y1, y2], color='g', linewidth=2)
                print(x1, x2)
            plt.show()
            _ = 1

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
            
    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)
        # The state is initialized to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False
        self.grid = MyDroneEval.RecursiveGrid()
        self.last_ts = 0

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()
         # Compute its position and set the following attribute
        self.lidar_val = savitzky_golay(self.lidar_values(), 21, 3)
        
        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
            self.paths = astar(self.grid.graph, self.grid.last_visited_node, self.grid.rescue_point)
            self.path_index = 0
            print("paths", self.paths)

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        #print("state: {}, can_grasp: {}, grasped entities: {}".format(self.state.name,
        #                                                              self.base.grasper.can_grasp,
         #                                                             self.base.grasper.grasped_entities))

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a rescue center or wounded person is detected, we use a special command
        ##########
        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_random()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1
            
        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.back_rescue(self.paths)
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        #update the grid
        gps = self.measured_gps_position()
       
        if gps not in self.grid:
            compass = convert_angle(self.measured_compass_angle()) # Returns values in [0, 180]
            self.grid.update(gps, compass, self.lidar_val)
        else:
            compass = convert_angle(self.measured_compass_angle())

            # Check if the GPS is in a different cell than the last visited waypoint
            current_cell = self.grid.get_cell_for_point(gps)
            if getattr(self, 'last_visited_node', None) is not None:
                print(current_cell)
                last_visited_waypoint_cell = self.grid.get_cell_for_point(self.last_visited_node.gps_coord)
                print(last_visited_waypoint_cell)
                if last_visited_waypoint_cell is not None and (current_cell['waypoint_x'] != last_visited_waypoint_cell['waypoint_x'] or current_cell['waypoint_y'] != last_visited_waypoint_cell['waypoint_y']):
                    # Update the last visited waypoint
                    self.last_visited_node = self.grid.graph.get_node_by_coords(self.grid.get_waypoint_for_cell(current_cell))
                    print("last visited", self.last_visited_node)

        return command

    def process_lidar_sensor(self, the_lidar_sensor):
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller = 0.5

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return command, False

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution

        far_angle_raw = 0
        near_angle_raw = 0
        min_dist = 1000
        if size != 0:
            # far_angle_raw : angle with the longer distance
            # 这两个角度没看明白，是什么意思？
            far_angle_raw = ray_angles[np.argmax(values)]
            min_dist = min(values)
            # near_angle_raw : angle with the nearest distance
            near_angle_raw = ray_angles[np.argmin(values)]

        far_angle = far_angle_raw
        # If far_angle_raw is small then far_angle = 0
        if abs(far_angle) < 1 / 180 * np.pi:
            far_angle = 0.0

        near_angle = near_angle_raw
        #print(f"near angle: {near_angle}")
        far_angle = normalize_angle(far_angle)
        #print(f"far angle: {far_angle}")

        # The drone will turn toward the zone with the more space ahead
        #控制往哪个方向转，顺时针转还是逆时针转
        if size != 0:
            if far_angle > 0:
                command["rotation"] = angular_vel_controller
            elif far_angle == 0:
                command["rotation"] = 0
            else:
                command["rotation"] = -angular_vel_controller

        # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
        # size是什么
        collision = False
        if size != 0 and min_dist < 10:
            collision = True
            if near_angle > 0:
                command["rotation"] = -angular_vel_controller
            else:
                command["rotation"] = angular_vel_controller

        return command, collision

    def control_random(self):
        """
        Here we change it and combine it with the lidar_communication
        """
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        command_lidar, collision_lidar = self.process_lidar_sensor(self.lidar())

        alpha = 0.4
        alpha_rot = 0.75

        if collision_lidar:
            alpha_rot = 0.1

        # The final command  is a combination of 2 commands
        command["forward"] = (1 - alpha) * command_lidar["forward"]
        command["lateral"] = (1 - alpha) * command_lidar["lateral"]
        command["rotation"] = (1 - alpha_rot) * command_lidar["rotation"]
        
        return command
    
    def back_rescue(self, paths):
        """
        Follows a predefined path while avoiding collisions using lidar data.
        """
        #print("back")
        command = {"forward": 0.0,
                "lateral": 0.0,
                "rotation": 0.0,
                "grasper": 0}

        command_lidar, collision_lidar = self.process_lidar_sensor(self.lidar())

        alpha = 0.4
        alpha_rot = 0.75

        if collision_lidar:
            alpha_rot = 0.1

        # The final command is a combination of lidar-based control and path-following control
        command_ret = self.follow_path(paths)
        command = command_ret
        
        return command

    def follow_path(self, paths):
        """
        Follows the predefined path using proportional control.
        """
        command = {"forward": 0.0,
                "lateral": 0.0,
                "rotation": 0.0,
                "grasper": 0}
        gps = self.measured_gps_position()
        compass = self.measured_compass_angle()

        # Assuming paths is a list of nodes (waypoints) to follow
        target_node = paths[self.path_index]
        # Obtain the angle to turn from current orientation
        alpha = np.arctan2(target_node.gps_coord[1] - gps[1], target_node.gps_coord[0] - gps[0]) - compass

        rot = min(np.exp(abs(alpha) / 90), 1)
        command["rotation"] = 0.7 * np.sign(alpha) * rot * gomperz(self.last_ts)
        command["forward"] = 0.4 * 0.3 if abs(rot) < 0.1 else 0.7
        command["lateral"] = 0.4 * - np.sign(alpha) * rot ** 2
        self.last_ts += 0.5

        # Check if the drone is close to the current target node
        distance_to_target = np.linalg.norm(np.array(gps) - np.array(target_node.gps_coord))
        if distance_to_target < 10.0:  # Adjust this threshold as needed
            self.path_index += 1  # Move to the next node in the path
            self.last_ts = 0
            print("index->", self.path_index)

        return command

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 50)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.5, 1)

        return found_wounded, found_rescue_center, command



class MyMapSemantic(MapAbstract):
    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (400, 400)

        self._rescue_center = RescueCenter(size=(100, 100))
        self._rescue_center_pos = ((0, 150), 0)

        # WOUNDED PERSONS
        self._number_wounded_persons = 20
        self._wounded_persons_pos = []
        self._wounded_persons: List[WoundedPerson] = []

        start_area = (0.0, -30.0)
        nb_per_side = math.ceil(math.sqrt(float(self._number_wounded_persons)))
        dist_inter_wounded = 60.0
        sx = start_area[0] - (nb_per_side - 1) * 0.5 * dist_inter_wounded
        sy = start_area[1] - (nb_per_side - 1) * 0.5 * dist_inter_wounded

        for i in range(self._number_wounded_persons):
            x = sx + (float(i) % nb_per_side) * dist_inter_wounded
            y = sy + math.floor(float(i) / nb_per_side) * dist_inter_wounded
            pos = ((x, y), random.uniform(-math.pi, math.pi))
            self._wounded_persons_pos.append(pos)

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        self._drones_pos = [((40, 40), random.uniform(-math.pi, math.pi))]
        self._drones = []

    def construct_playground(self, drone_type: Type[DroneAbstract]):
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)

        playground.add(self._rescue_center, self._rescue_center_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = self._wounded_persons_pos[i]
            playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMapSemantic()
    playground = my_map.construct_playground(drone_type=MyDroneEval)

    # draw_semantic_rays : enable the visualization of the semantic rays
    gui = GuiSR(playground=playground,
                the_map=my_map,
                draw_semantic_rays=True,
                use_keyboard=False,
                )
    gui.run()


if __name__ == '__main__':
    main()