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
    if angle < 0:
        return int(np.round(180 + 180 * angle / (2*np.pi)))
    return int(np.round(180 * angle / (2*np.pi)))

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
    class Waypoint:
        x : float
        y : float
        forward : list
        backward : list
        directions : np.array

        def __init__(self, pos, ):
            self.x, self.y = pos
            self.forward = []
            self.backward = []
            self.directions = np.zeros((180,))
            
        def distance(self, pos):
            return np.sqrt((self.x - pos[0])**2 + (self.y - pos[1])**2)
        
        def orientation(self, pos):
            return np.arctan((self.y - pos[1]) / (self.x - pos[0])) if np.abs(self.x - pos[0]) >= 1e-3 else 0
        
        def add(self, other):
            self.forward.append(other)
            other.backward.append(self)
            
        def add_directions(self, directions):
            directions = np.delete(directions, np.isin(directions, np.where(self.directions == -1)[0]))
            self.directions[directions] = 1
        
        def choose(self, p):
            dirs = np.where(self.directions > 0)[0]
            if len(dirs) == 0:
                dirs = np.where(self.directions == 0)[0]
                return random.choice(dirs)
            d = random.choices(dirs, weights=self.directions[dirs])[0]
            self.directions *= p
            self.directions[d] /= p**2
            return d
        
        def delete(self, dir):
            # Redistribute the weight among all the other directions
            n = np.sum(self.directions > 0) - 1
            if n > 0:
                self.directions[self.directions > 0] += self.directions[dir]/n
            self.directions[dir] = -1
            
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

        self.v_n_sq = 0
        self.alpha = 0
        self.w = 0

        self.waypoint = None
        self.c = 0  # only for debug
        random.seed(identifier)
        self.p = random.randint(1, 10) / 100
        self.state = MyDroneEval.States.SEARCHING
        self.collision = False
        self.body = None
        self.following = None
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
        
        if self.lidar_val[90-22:90+22].min() < d:  # FOV: 180 degrees
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
            elif self.state == MyDroneEval.States.FOUND_BODY:
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
        gps = self.measured_gps_position()
        compass_180 = convert_angle(self.measured_compass_angle())
        wounded = self.poi[self.poi["Type"]==MyDroneEval.Wounded]
        if not wounded.empty:
            # A person to rescue is found in the bank
            wounded = wounded[wounded['Obj'].apply(lambda p: p.distance(gps)).argmin()]
            self.body = wounded
            self.state = MyDroneEval.States.FOUND_BODY
            self.angle_stop = 2*np.arctan((wounded.y - gps[1]) / (wounded.x - gps[0]))
            self.rotation = np.sign(self.angle_stop)
        else:
            # Searching for people
            if self.waypoint is None:
                self.waypoint = MyDroneEval.Waypoint(gps)
                self.following = self.waypoint

            # Add new waypoint if needed
            corner = np.abs(savitzky_golay(self.lidar_val, 11, 3, deriv=1))
            # if corner.max() > 15 and self.waypoint.distance(gps) > 15:
                # Extract one of the directions of rapid change for the lidar
                # i = np.where(corner > 0.80 * corner.max())[0]
                # Compute the lidar ray relative to the actual waypoint
                # waypoint_alpha = convert_angle(2*self.waypoint.orientation(gps))
                # if compass_180 - waypoint_alpha < 90:
                #     waypoint_alpha = 90 + compass_180 - waypoint_alpha
                # else:
                #     waypoint_alpha = 90 + 180 - (compass_180 - waypoint_alpha)
                # if 0.8 < self.waypoint.distance(gps) / self.lidar_val[waypoint_alpha] < 1:
                # w = self.waypoint
                # self.waypoint = MyDroneEval.Waypoint(gps)
                # w.add(self.waypoint)
                

                # express the angles sp that they start from the horizontal angle at global 0 degrees
                # formula to pass from global reference system to lidar reference system:
                # alpha_GRS = alpha_LRS + alpha_GRS2LRS = alpha_LRS + compass_180 - 90
                # alpha = np.array([a - 90 + compass_180 if a - 90 + compass_180 < 180 else a - 90 + compass_180 - 180 for a in i]) # lidar first position is -pi
                # self.waypoint.add_directions(alpha)

            # If we reached the destination find another one
            if 1 < self.following.distance(gps) < 10:
                angle = self.waypoint.choose(1 + self.p)
                angle = np.array([np.cos(angle), np.sin(angle)])
                r = np.random.rand(1,) * self.lidar_val.min()
                self.following = MyDroneEval.Waypoint(gps + r*angle)
        
        self.reach()
        
    
    def reach(self):
        """Reaches the entity passed as parameter. Whether it is a body, the recovery center or other"""
        gps = self.measured_gps_position()
        # compass = self.measured_compass_angle()
        
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
            
            