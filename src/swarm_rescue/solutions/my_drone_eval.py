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

def convert_angle(angle):
    return int(np.round(90 * (angle + np.pi) / np.pi))

def entity_mapping(type: DroneSemanticSensor.TypeEntity):
    if type == DroneSemanticSensor.TypeEntity.WALL:
        return MyDroneEval.Wall
    elif type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
        return MyDroneEval.Wounded
    elif type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
        return MyDroneEval.RescueZone
    # Updating a drone may be not useful. Could do it with communication system instead
    # elif type == DroneSemanticSensor.TypeEntity.DRONE:
    #     return "Drone"

def separate(v):
    start = 0
    for i in range(1, len(v)):
        if not (v[i-1][3]==v[i][3] and v[i-1][4]==v[i][4]) and (v[i][3]==v[min(i, i+1)][3] and v[i][4]==v[min(i, i+1)][4]):
            yield v[start:i]
            start = i
    if start == 0:
        return v

def instantiate_entity(cl, v):
    obj = cl()
    for p in separate(v):
        arr = np.array(p)
        dim = arr[:2].var(axis=0).argmin()
        n = arr[:, 2].sum()
        
        if cl == MyDroneEval.Wounded:
            obj.x = arr[0].mean()
            obj.y = arr[1].mean()
            obj.xconf = n
            obj.yconf = n
        elif cl == MyDroneEval.Wall:
            if dim == 0:
                obj.xlim = arr[:, dim].mean()
                obj.xconf = n
            else:
                obj.ylim = arr[:, dim].mean()
                obj.yconf = n
        elif cl == MyDroneEval.Zone:
            if dim == 0:
                rightlim = arr[-1][4]
                if rightlim:
                    obj.xlim = (arr[:, dim].mean(), np.inf)
                    obj.xconf = n
                else:
                    obj.xlim = (-np.inf, arr[:, dim].mean())
                    obj.xconf = n
            else:
                uplim = arr[-1][3]
                if uplim:
                    obj.ylim = (arr[:, dim].mean(), np.inf)
                    obj.yconf = n
                else:
                    obj.ylim = (-np.inf, arr[:, dim].mean())
                    obj.yconf = n
    return obj

class MyDroneEval(DroneAbstract):
    class RecursiveGrid:
        cells: pd.DataFrame
        
        def __init__(self) -> None:
            self.cells = pd.DataFrame({"ulim": [], "llim": [], "dlim": [], "rlim": [], "waypoint_x": [], "waypoint_y": []})
            
        def update(self, gps_coord, compass_angle, rays):
            rays = np.roll(rays[:-1], compass_angle)
            lims = [0]*4    # limit in the 4 directions
            already_set_limits = self.cells.loc[:, "ulim":"rlim"] - np.array([gps_coord[1], gps_coord[0], gps_coord[1], gps_coord[0]])

            up = rays[0:90]
            up_wall = up * np.sin(np.pi*np.arange(90)/180)
            up_lim = min(0.75*up_wall.max(), already_set_limits.loc[already_set_limits["dlim"] > 0, "dlim"].min())
            lims[0] = gps_coord[1] + up_lim

            left = rays[45:135]
            left_wall = left * np.sin(np.pi*np.arange(90)/180)
            left_lim = min(0.75*left_wall.max(), -already_set_limits.loc[already_set_limits["rlim"] < 0, "rlim"].max()) # -max(x) = min(-x)
            lims[1] = gps_coord[0] - left_lim

            down = rays[90:]
            down_wall = down * np.sin(np.pi*np.arange(90)/180)
            down_lim = min(0.75*down_wall.max(), -already_set_limits.loc[already_set_limits["ulim"] < 0, "ulim"].max())
            lims[2] = gps_coord[1] - down_lim

            right = np.concatenate((rays[-45:], rays[:45]))
            right_wall = right * np.sin(np.pi*np.arange(90)/180)
            right_lim = min(0.75*right_wall.max(), already_set_limits.loc[already_set_limits["llim"] > 0, "llim"].min())
            lims[3] = gps_coord[0] + right_lim
            
            lims += [(lims[0] + lims[2])/2, (lims[1] + lims[3])/2]
            self.cells.loc[-1] = lims
            self.cells.index += 1
        
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
            return (self.x - x)**2 + (self.y - y)
            
    class States(Enum):
        SEARCHING = 0
        FOUND_BODY = 1
        CARRYING_BODY = 2
        
    class Wounded:
        x: float = np.nan
        y: float = np.nan
        xconf: int = 0
        yconf: int = 0
        grasped: bool = False
        due_to: int = -1
        
        def distance(self, robot):
            return (self.x-robot[0])**2 + (self.y - robot[1])**2
    
    class Wall:
        xlim: float | Tuple[float, float] = np.nan
        ylim: float | Tuple[float, float] = np.nan
        xconf: int = 0
        yconf: int = 0
        
    class Zone:
        xlim: Tuple[float, float] = (-np.inf, np.inf)
        ylim: Tuple[float, float] = (-np.inf, np.inf)
        xconf: Tuple[int, int] = (0, 0)
        yconf: Tuple[int, int] = (0, 0)
        type: str = "Generic"
        
    class RescueZone(Zone):
        type = "rescue"
        
    class DestructionZone(Zone):
        type = "destruction"
        
    class NoGPSZone(Zone):
        type = "nogps"
    
    class NoCommZone(Zone):
        type = "nocomm"
        
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


        random.seed(identifier)
        self.grid = MyDroneEval.RecursiveGrid()
        self.state = MyDroneEval.States.SEARCHING
        self.body = None
        self.following = MyDroneEval.ReachWrapper((np.inf, np.inf))
        self.poi = pd.DataFrame(np.zeros((1, 3)), columns=["Obj", "Id", "Type"], dtype=np.float32)
        self.ids = {MyDroneEval.Wall: 0, MyDroneEval.Wounded: 0, MyDroneEval.RescueZone: 0, MyDroneEval.NoGPSZone: 0, MyDroneEval.NoCommZone: 0}
    
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
        self.estimate_objects()
        self.lidar_val = savitzky_golay(self.lidar_values(), 21, 3)
        
        if self.lidar_val[90-22:90+22].min() < d:  # FOV: 90 degrees
            self.collision = True
            self.solve_collision()
        else:
            if self.state == MyDroneEval.States.SEARCHING:
                # if self.following is None:
                self.search()
                # else:
                #     self.reach()
            elif self.state == MyDroneEval.States.FOUND_BODY:
                self.reach(self.body)
            elif self.state == MyDroneEval.States.CARRYING_BODY:
                rescue = self.poi[self.poi["Type"] == MyDroneEval.RescueZone]
                self.reach(rescue["Obj"])


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
        compass = convert_angle(self.measured_compass_angle()) # Returns values in [-pi, pi]
        wounded = self.poi[self.poi["Type"]==MyDroneEval.Wounded]
        # If a new cell can be created and its middle point can be assigned add it to the DataFrame
        if self.following.distance(gps) < 1e-1:
            self.grid.update(gps, compass, self.lidar_val)
        if not wounded.empty:
            # A person to rescue is found in the bank
            wounded = wounded[wounded['Obj'].apply(lambda p: p.distance(gps)).argmin()]
            self.body = wounded
            self.state = MyDroneEval.States.FOUND_BODY
            self.angle_stop = 2*np.arctan((wounded.y - gps[1]) / (wounded.x - gps[0]))
            self.rotation = np.sign(self.angle_stop)
        else:
            # Check with the cells with neighboring ids and move to unseen scenarios
            rays = np.roll(self.lidar_val[:-1], 180-compass)
            q80 = np.quantile(self.lidar_val, 0.8)
            guess = gps + (rays * np.vstack([np.cos(np.arange(0, 2*np.pi, 2*np.pi/180)),
                                                        np.sin(np.arange(0, 2*np.pi, 2*np.pi/180))])).T
            guess = guess[rays > q80]
            guess = np.apply_along_axis(lambda row: row if row not in self.grid else np.array([np.nan, np.nan]), 1, guess)
            guess = guess[(~np.isnan(guess)).any(axis=1)]

            if guess.size == 0:
                self.following = MyDroneEval.ReachWrapper(self.grid.cells.loc[1, ["waypoint_x", "waypoint_y"]])
            else:
                self.following = MyDroneEval.ReachWrapper(guess[np.random.choice(guess.shape[0])])
        
        self.reach()
        
    
    def reach(self):
        """Reaches the entity defined in self.following."""
        gps = self.measured_gps_position()
        
        alpha = np.arctan2((self.following.y - gps[1]) , (self.following.x - gps[0]))

        N = 11
        R = (d - 25) / (1 - np.sin(alpha))
        w = (np.pi/2 - alpha) / N
        
        delta_x = self.odometer_values()[0]
        acc = max(-1, min(1, 4*np.sin(np.pi / N)*R - 2*delta_x))
        
        self.forward = acc*np.cos(w)
        self.lateral = acc*np.sin(w)
        self.rotation = w
    
    def estimate_objects(self):
        detection_semantic = self.semantic_values()
        gps = self.measured_gps_position()
        compass = self.measured_compass_angle()
        
        estimates = defaultdict(list)
        
        for data in detection_semantic:
            # Compute the position of the object
            alpha = data.angle
            d = data.distance
            x, y = gps[0]+d*np.cos(alpha+compass), gps[1]+d*np.sin(alpha+compass)
            up = alpha + compass > 0
            right = 0< alpha + compass + np.pi/2 < np.pi

            entity_type = entity_mapping(data.entity_type)
            estimates[entity_type].append((x, y, 1, up, right))

        for k, v in estimates.items():
            # Update old value with the new data retrieved
            obj = instantiate_entity(k, v)
            
            self.poi["Obj"] = obj
            self.poi["Id"] = self.ids[k]
            self.poi["Type"] = k
            self.ids[k] += 1
            
            