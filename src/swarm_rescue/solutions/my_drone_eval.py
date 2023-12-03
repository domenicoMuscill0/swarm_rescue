from collections import defaultdict
from enum import Enum
import math
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor

import matplotlib.pyplot as plt

d = 40
l = 20
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

def convert_angle(angle):
    if angle >= 0:
        return int(np.round(90 * angle / np.pi))
    else:
        return int(np.round(180 + 90 * angle / np.pi))
    
def clean_material_following(rays, semantic):
    for s in semantic:
        if s.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
            body_ray = (convert_angle(s.angle)+90)%180
            rays[body_ray-5:body_ray+5] = np.inf
    return rays



class MyDroneEval(DroneAbstract):
    class RecursiveGrid:
        cells: pd.DataFrame
        
        def __init__(self) -> None:
            self.cells = pd.DataFrame({"ulim": [], "llim": [], "dlim": [], "rlim": [], "waypoint_x": [], "waypoint_y": []})
            
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

            plt.show()
            _ = 1
        
        def __contains__(self, point):
            return np.any((self.cells['llim'] < point[0]) & (point[0] <= self.cells['rlim']) & 
                          (self.cells['dlim'] < point[1]) & (point[1] <= self.cells['ulim']))
        
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
            
    class States(Enum):
        SEARCHING = 0
        FOUND_BODY = 1
        CARRYING_BODY = 2
        
        
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        self.forward = 0
        self.lateral = 0
        self.rotation = 0
        self.grasper = 0
        self.last_ts = 0


        random.seed(identifier)
        self.grid = MyDroneEval.RecursiveGrid()
        self.state = MyDroneEval.States.SEARCHING
        self.following = MyDroneEval.ReachWrapper((np.inf, np.inf))
    
    def define_message_for_all(self):
        """No comunication for now.
           Should:
           - Send the request for the recovery center if not in self.poi.
           - Share positions and distance to wounded people nearby so to schedule which drones
           is due to take them.
           - On Wall discovery send the new id so that global id is updated. 
           - On request for body organize with others robots nearby
           """
        pass

    def control(self):
        
        # TODO : put this in a thread
        # Check if there is a visible body or the rescue center
        semantic = self.semantic_values()
        gps = self.measured_gps_position()
        bodies = [ray for ray in semantic if ray.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not ray.grasped]
        if len(bodies) > 0 and self.state != MyDroneEval.States.CARRYING_BODY:
            compass = self.measured_compass_angle()
            positions = [(np.cos(body.angle - compass)*body.distance, np.sin(body.angle - compass)*body.distance) for body in bodies]
            x, y = zip(*positions)
            x, y = sum(x) / len(x), sum(y) / len(y)
            self.following = MyDroneEval.ReachWrapper((gps[0]+x, gps[1]+y))
            self.state = MyDroneEval.States.FOUND_BODY
            self.last_ts = 0

        # Compute its position and set the following attribute
        self.lidar_val = savitzky_golay(self.lidar_values(), 21, 3)
        
        if clean_material_following(self.lidar_val, semantic)[90-22:90+22].min() < d:  # FOV: 90 degrees
            self.solve_collision()
        else:
            if self.state == MyDroneEval.States.SEARCHING:
                # if self.following is None:
                self.search()
                # else:
                #     self.reach()
            elif self.state == MyDroneEval.States.FOUND_BODY:
                print("AOOOOOOOOOOOOOO", self.following.distance(self.measured_gps_position()))
                if self.following.distance(self.measured_gps_position()) < 30:
                    self.grasper = 1
                self.reach()
            elif self.state == MyDroneEval.States.CARRYING_BODY:
                self.grasper = 1
                # self.following = rescue
                self.reach()


        return {"forward": self.forward,
                "lateral": self.lateral,
                "rotation": self.rotation,
                "grasper": self.grasper}
    
    def solve_collision(self):
        """If there is a collision start turning towards the most secure direction and then start accelerating"""
        # Computed approximating the path that it has to follow
        # on the border of a regular N-polygon
        
        alpha = np.pi * np.argmin(self.lidar_val[90-45:90+45]) / 90
        l = np.min(self.lidar_val)
        N = 5
        # R is the radius of the osculatrix circumference 
        R = l / (1 - np.sin(alpha))
        # W is the angular velocity
        w = (np.pi/2 - alpha) / N
        
        delta_x = self.odometer_values()[0]
        acc = max(-1, min(1, 4*np.sin(np.pi / N)*R - 2*delta_x))
        
        self.forward = np.sign(d/2-R)*acc*np.cos(w)
        self.lateral = np.sign(d/2-R)*acc*np.sin(w)
        self.rotation = w
    
    def search(self):
        """Logic for determining the next goal position in the map"""
        gps = self.measured_gps_position()
        compass = convert_angle(self.measured_compass_angle()) # Returns values in [0, 180]
        goal_angle = convert_angle(np.arctan2(self.following.y - gps[1], self.following.x - gps[0]))
        d2goal = self.lidar_val[(90+goal_angle-compass)%181]
        # If a new cell can be created and its middle point can be assigned add it to the DataFrame
        print(self.following.distance(gps))
        if not (d < self.following.distance(gps) < np.inf) or self.following.distance(gps) / d2goal > 1.02:
            if gps not in self.grid:
                self.grid.update(gps, compass, self.lidar_val)
            # Check with the cells with neighboring ids and move to unseen scenarios
            rays = np.roll(self.lidar_val[:-1], 90+compass)
            q80 = np.quantile(self.lidar_val, 0.8)
            guess = gps + (rays * np.vstack([np.cos(np.arange(0, 2*np.pi, 2*np.pi/180)),
                                                        np.sin(np.arange(0, 2*np.pi, 2*np.pi/180))])).T
            guess = guess[rays > q80]
            guess = np.apply_along_axis(lambda row: row if row not in self.grid else np.array([np.nan, np.nan]), 1, guess)
            guess = guess[(~np.isnan(guess)).any(axis=1)]

            if guess.size == 0:
                self.following = MyDroneEval.ReachWrapper(self.grid.cells.loc[1, ["waypoint_x", "waypoint_y"]])
                print([self.following.x, self.following.y])
            else:
                self.following = MyDroneEval.ReachWrapper(guess[np.random.choice(guess.shape[0])])
                print([self.following.x, self.following.y])
            self.last_ts = 0
        
        self.reach()
        
    
    def reach(self):
        """Reaches the entity defined in self.following."""
        # gps = self.measured_gps_position()
        # compass = self.measured_compass_angle()
        # dist = self.following.distance(gps)
        
        # alpha = np.arctan2(self.following.y - gps[1], self.following.x - gps[0]) # - compass

        # R = l / (1 - np.sin(alpha))
        # N = max(11, np.pi / (1e-2 + math.asin(l / (2*R))))
        # w = (np.pi/2 - alpha) / N
        
        # delta_x = self.odometer_values()[0]
        # acc = max(-1, min(1, 4*np.sin(np.pi / N)*R - 2*delta_x))
        
        # self.forward = acc*np.cos(w)
        # self.lateral = acc*np.sin(w)
        # self.rotation = w if abs(alpha - compass) >= 0.5 else 0
        # self.inertia[0] += self.forward
        # self.inertia[1] += self.lateral
        gps = self.measured_gps_position()
        compass = self.measured_compass_angle()
        # Obtain the angle to turn from current orientation
        alpha = np.arctan2(self.following.y - gps[1], self.following.x - gps[0]) - compass

        rot = min(np.exp(abs(alpha) / 90), 1)
        self.rotation = np.sign(alpha)*rot*gomperz(self.last_ts)
        self.forward = 0.5 if abs(rot) < 0.1 else 1
        self.lateral = np.sign(alpha)*rot**2
        self.last_ts += 1
