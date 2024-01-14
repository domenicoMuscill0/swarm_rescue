from Classes.maze import Maze
from Classes.cam import Detector
from Classes.robot import Robot 
import math
import random
import time
import threading
import numpy as np
import cv2 as cv
import pickle

# Here is the idea of building the global map
def building_global_maze(dim):
    global_maze = Maze(dim, dim)
    global_maze.build_maze()
    File = open("mmm.pickle",'wb')
    File.truncate()
    pickle.dump(global_maze, File)
    File.close()
    return global_maze 

def build_robots(k, dim):
    maze = Maze(dim, dim)
    maze.build()
    goal_cell = maze.generate_random_cell()
    robots = []
    for i in range(k):
        robots.append(Robot(10+i, colors[i], maze, goal_cell))
    maze.add_robots(robots)
    return robots, maze

# I haven't figure out what is the pickle? What does it mean?
def import_global_maze():
    File = open("mmm.pickle",'rb')
    global_maze = pickle.load(File)
    File.close()
    return global_maze

def build_cam():
    robot_cam = Detector(maze, "robots", lock, robots)
    global_cam = Detector(global_maze, "global", lock)
    for i in range(k):
        robots[i].add_robot_cam(robot_cam)
    return robot_cam, global_cam

def main(global_maze):
    t = []
    for i in range(k):
        t.append(threading.Thread(target = robots[i].maze_exploration,
                                  args = (maze, global_maze, robot_cam, lock)))
    for i in range(len(t)):
        t[i].start()
    for i in range(len(t)):
        t[i].join()
    
    for i in range(k):
        robots[i].add_global_maze(global_maze)
        robots[i].goal_cell = maze.return_cell((10,1))
        t.append(threading.Thread(target = robots[i].follow_path, args = 
                                  (paths[i], maze, lock)))
    for i in range(len(t)):
        t[i].start()
    for i in range(len(t)):
        t[i].join()

if __name__ == '__main__':
    infile = open("dict", "rb")
    dic = pickle.load(infile)
    infile.close()
    for dim in [10, 15, 17, 20, 25]:
        for k in [1, 3, 5, 7]:
            global_maze = build_global_maze(dim)
            robots, maze = build_robots(k, dim)
            lock = threading.Lock()
            robot_cam, global_cam = build_cam()
            main(global_maze)
            max_overall = 0
            max_phase_one = 0
            for robot in robots:
                max_phase_one = max(robot.phase_one_counter, max_phase_one)
                max_overall = max(robot.overall_counter, max_overall)
            dic[k][dim].append((max_phase_one, max_overall))
            outfile = open("dict", "web")
            pickle.dump(dic, outfile)
            outfile.close()
            
