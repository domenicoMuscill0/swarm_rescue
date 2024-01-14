import random
import time
import datetime import datetime
from collections import defaultdict
import math
import sys
from dijkstar import Graph, find_path

def euc_dist(cell1, cell2):
    return math.sqrt((cell1.Row - cell2.Row)**2 + (cell1.Col - cell2.Col)**2)

class Robot:
    def __init__(self, color, rgb, main_maze, goal_cell, initial_pos = None):
        self.Xpos = 0
        self.Ypos = 0
        self.RowInd = 0
        self.Front = 1
        self.Left = 1
        self.Right = 1
        self.Back = 1
        self.direction = 0 # 0 left, 1 up, 2 right, 3 down
        self.visited = defaultdict(int)
        self.color = color
        self.rgb = rgb
        self.goal_cell = goal_cell
        self.initializePos(main_maze, initial_pos)
        self.WL =defaultdict(float)
        self.key = "sum_of_distance"
        self.destination = None
        self.next_Cell = None
        self.changed = " "
        self.phase_one_counter = 0
        self.overall_counter = 0

def CurrentCell(self, maze):
    return maze.cell[self.RowInd][self.ColInd]

def CurrentCell(self, maze):
    return maze.cell[self.RowInd][self.ColInd]

def LeftCell(self, maze):
    if self.direction == 0:  # left
        return maze.downCell(self.CurrentCell(maze))
    elif self.direction == 1:  # up
        return maze.leftCell(self.CurrentCell(maze))
    elif self.direction == 2:  # right
        return maze.upCell(self.CurrentCell(maze))
    elif self.direction == 3:  # down
        return maze.rightCell(self.CurrentCell(maze))

def RightCell(self, maze):
    if self.direction == 0:  # left
        return maze.upCell(self.CurrentCell(maze))
    elif self.direction == 1:  # up
        return maze.rightCell(self.CurrentCell(maze))
    elif self.direction == 2:  # right
        return maze.downCell(self.CurrentCell(maze))
    elif self.direction == 3:  # down
        return maze.leftCell(self.CurrentCell(maze))

def BackCell(self, maze):
    if self.direction == 0:  # left
        return maze.rightCell(self.CurrentCell(maze))
    elif self.direction == 1:  # up
        return maze.downCell(self.CurrentCell(maze))
    elif self.direction == 2:  # right
        return maze.leftCell(self.CurrentCell(maze))
    elif self.direction == 3:  # down
        return maze.upCell(self.CurrentCell(maze))

def FrontCell(self, maze):
    if self.direction == 0:  # left
        return maze.leftCell(self.CurrentCell(maze))
    elif self.direction == 1:  # up
        return maze.upCell(self.CurrentCell(maze))
    elif self.direction == 2:  # right
        return maze.rightCell(self.CurrentCell(maze))
    elif self.direction == 3:  # down
        return maze.downCell(self.CurrentCell(maze))

def FrontWall(self, maze):
    if self.direction == 0:  # left
        return self.FrontCell(maze).rightWall
    elif self.direction == 1:  # up
        return self.FrontCell(maze).downWall
    elif self.direction == 2:  # right
        return self.CurrentCell(maze).rightWall
    elif self.direction == 3:  # down
        return self.CurrentCell(maze).downWall

def BackWall(self, maze):
    if self.direction == 0:  # left
        return self.CurrentCell(maze).rightWall
    elif self.direction == 1:  # up
        return self.CurrentCell(maze).downWall
    elif self.direction == 2:  # right
        return self.BackCell(maze).rightWall
    elif self.direction == 3:  # down
        return self.BackCell(maze).downWall

def LeftWall(self, maze):
    if self.direction == 0:  # left
        return self.CurrentCell(maze).downWall
    elif self.direction == 1:  # up
        return self.LeftCell(maze).rightWall
    elif self.direction == 2:  # right
        return self.LeftCell(maze).downWall
    elif self.direction == 3:  # down
        return self.CurrentCell(maze).rightWall

def RightWall(self, maze):
    if self.direction == 0:  # left
        return self.RightCell(maze).downWall
    elif self.direction == 1:  # up
        return self.CurrentCell(maze).rightWall
    elif self.direction == 2:  # right
        return self.CurrentCell(maze).downWall
    elif self.direction == 3:  # down
        return self.RightCell(maze).rightWall

