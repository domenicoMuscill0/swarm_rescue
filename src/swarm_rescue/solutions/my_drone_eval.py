from collections import defaultdict
from enum import Enum
import math
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from solutions.my_drone_random import MyDroneRandom
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor


def entity_mapping(type: DroneSemanticSensor.TypeEntity):
    if type == DroneSemanticSensor.TypeEntity.WALL:
        return MyDroneEval.Wall
    elif type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
        return MyDroneEval.Wounded
    elif type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
        return MyDroneEval.Zone
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
        
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        self.state = MyDroneEval.States.SEARCHING
        self.angle_stop = None
        self.body = None
        self.poi = pd.DataFrame(np.zeros((1, 3)), columns=["Obj", "Id", "Type"], dtype=np.float32)
        self.ids = {MyDroneEval.Wall: 0, MyDroneEval.Wounded: 0, MyDroneEval.Zone: 0}
    
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
        command = {"forward": 0.1,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        # TODO : put this in a thread
        self.estimate_objects()

        if self.angle_stop is None:
            self.angle_stop = self.measured_compass_angle()

        if self.state == MyDroneEval.States.SEARCHING:
            command = self.search()
            
        if self.state == MyDroneEval.States.FOUND_BODY:
            command = self.reach(self.body)

        # lidar = self.lidar_values()
        # closest_deg_idx = lidar.argmin()
        # closest_deg = -3/2 + (closest_deg_idx - 1) / 90

        
        # compass = self.measured_compass_angle()
        # # We go to the right by default
        # if 0 < closest_deg < 1/2:
        #     command["rotation"] = -1/2 - closest_deg
        # elif closest_deg >= 1/2:
        #     command["rotation"] = closest_deg - 3/2
        # else:
        #     command["rotation"] = 1/2 + closest_deg

        return command
    
    def search(self):
        gps = self.measured_gps_position()
        command = {"forward": 0.1,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        wounded = self.poi[self.poi["Type"]==MyDroneEval.Wounded]
        if not wounded.empty:
            wounded = wounded[wounded.apply(lambda p: p.distance(gps)).argmin()]
            self.body = wounded
            self.state = MyDroneEval.States.FOUND_BODY
            self.angle_stop = 2*np.arctan((wounded.y - gps[1]) / (wounded.x - gps[0]))
            command["rotation"] = np.sign(self.angle_stop)
        elif abs(normalize_angle(self.angle_stop - self.measured_compass_angle())) < 0.2:
            lidar = self.lidar_values()
            self.angle_stop = np.pi*(-3/2 + (lidar.argmax() - 1) / 90)
            command["rotation"] = np.sign(self.angle_stop)
        else:
            command["forward"] = 1
        
        return command
    
    def reach(self, obj):
        """Reaches the entity passed as parameter. Whether it is a body, the recovery center or other"""
        pass
    
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
            
            