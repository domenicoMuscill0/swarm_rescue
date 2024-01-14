import random
import time
from datetime import datetime
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

def TurnLeft(self):
    self.direction = (self.direction - 1)%4

def TurnRight(self):
    self.direction = (self.direction + 1)%4

def ForwardOneCell(self, maze, speed = 2):
    front_cell = self.FrontCell(maze)
    self.change_pos_to(front_cell)

def change_pos_to(self, cell):
    self.RowInd = cell.Row
    self.ColInd = cell.Col
    self.Xpos = cell.xpos 
    self.Ypos = cell.ypos 

def initializePos(self, maze, initial_pos):
    random.seed(datetime.now())
    if initial_pos is None:
        first_cell = maze.generate_random_cell()
        while first_cell == self.goal_cell:
            first_cell = maze.generate_random_cell()
    else: first_cell = initial_pos
    if first_cell.color != 0:
        first_cell = maze.generate_random_cell()
    self.change_pos_to(first_cell)
    self.direction = random.randint(0, 3)
    self.CurrentCell(maze).set_color(self.color)
    self.CurrentCell(maze).set_OC_flag(True)
    self.add_to_visited_cell(self.CurrentCell(maze))

def add_to_visited_cell(self, next_Cell):
    self.visited[next_Cell.id] += 1

def explore(self, maze, global_maze, robot_cam, lock):
    self.AssignWall(maze, global_maze, robot_cam)
    next_Cell = self.Choose_direction(maze)
    self.Move(maze, next_Cell)

def Move(self, maze, next_Cell):
    self.overall_counter += 1
    if next_Cell == self.CurrentCell(maze):
        return

    prevCell = self.CurrentCell(maze)
    next_Cell.OC_flag = True

    if next_Cell == self.LeftCell(maze):
        self.TurnLeft()
    elif next_Cell == self.RightCell(maze):
        self.TurnRight()
    elif next_Cell == self.FrontCell(maze):
        pass
    elif next_Cell == self.BackCell(maze):
        self.TurnLeft()
        self.TurnLeft()

    self.color_assign(maze, next_Cell)
    self.ForwardOneCell(maze)

    prevCell.OC_flag = False
    self.add_to_visited_cell(next_Cell)

def color_assign(self, maze, next_Cell):
    # done
    if next_Cell.get_color() == 0:
        if self.direction_sum(maze) == 1:
            self.dead_end(maze)
        else:
            self.CurrentCell(maze).set_color(1)
            self.robot_cam.set_color(self.CurrentCell(maze), 1)
    elif next_Cell.get_color() == 1:
        if self.direction_sum(maze) == 1:
            self.dead_end(maze)
        else:
            self.CurrentCell(maze).set_color(1)
            self.robot_cam.set_color(self.CurrentCell(maze), 1)
    next_Cell.set_color(self.color)
    self.robot_cam.set_color(next_Cell, self.color, self)

def dead_end(self, maze):
    # done
    self.CurrentCell(maze).set_color(2)
    self.robot_cam.set_color(self.CurrentCell(maze), 2)
    maze.add_obstacle(self.CurrentCell(maze), maze.up_cell(self.CurrentCell(maze)), self.robot_cam)
    maze.add_obstacle(self.CurrentCell(maze), maze.left_cell(self.CurrentCell(maze)), self.robot_cam)
    maze.add_obstacle(self.CurrentCell(maze), maze.right_cell(self.CurrentCell(maze)), self.robot_cam)
    maze.add_obstacle(self.CurrentCell(maze), maze.down_cell(self.CurrentCell(maze)), self.robot_cam)

def direction_sum(self, maze):
    # done
    return self.FrontWall(maze) + self.BackWall(maze) + self.LeftWall(maze) + self.RightWall(maze)

def AssignWall(self, maze, global_maze, robot_cam):
    # done
    if self.LeftWall(global_maze) == 0 and self.LeftWall(maze) == 1:
        maze.add_obstacle(self.CurrentCell(maze), self.LeftCell(maze), robot_cam)
    elif self.LeftWall(global_maze) == 1:
        maze.add_to_graph(self.CurrentCell(maze), self.LeftCell(maze))

    if self.FrontWall(global_maze) == 0 and self.FrontWall(maze) == 1:
        maze.add_obstacle(self.CurrentCell(maze), self.FrontCell(maze), robot_cam)
    elif self.FrontWall(global_maze) == 1:
        maze.add_to_graph(self.CurrentCell(maze), self.FrontCell(maze))

    if self.RightWall(global_maze) == 0 and self.RightWall(maze) == 1:
        maze.add_obstacle(self.CurrentCell(maze), self.RightCell(maze), robot_cam)
    elif self.RightWall(global_maze) == 1:
        maze.add_to_graph(self.CurrentCell(maze), self.RightCell(maze))

    if self.BackWall(global_maze) == 0 and self.BackWall(maze) == 1:
        maze.add_obstacle(self.CurrentCell(maze), self.BackCell(maze), robot_cam)
    elif self.BackWall(global_maze) == 1:
        maze.add_to_graph(self.CurrentCell(maze), self.BackCell(maze))

    robot_cam.present()

def Cell_categorize(self, maze, next_Cell=None):
    # done
    white_cells = []
    grey_cells = []

    if self.LeftCell(maze) and self.LeftWall(maze) == 1 and self.LeftCell(maze).get_OC_flag() is False:
        if self.LeftCell(maze).get_color() == 0:
            white_cells.append(self.LeftCell(maze))
        if self.LeftCell(maze).get_color() == 1:
            grey_cells.append(self.LeftCell(maze))

    if self.FrontCell(maze) and self.FrontWall(maze) == 1 and self.FrontCell(maze).get_OC_flag() is False:
        if self.FrontCell(maze).get_color() == 0:
            white_cells.append(self.FrontCell(maze))
        if self.FrontCell(maze).get_color() == 1:
            grey_cells.append(self.FrontCell(maze))

    if self.RightCell(maze) and self.RightWall(maze) == 1 and self.RightCell(maze).get_OC_flag() is False:
        if self.RightCell(maze).get_color() == 0:
            white_cells.append(self.RightCell(maze))
        if self.RightCell(maze).get_color() == 1:
            grey_cells.append(self.RightCell(maze))

    if self.BackCell(maze) and self.BackWall(maze) == 1 and self.BackCell(maze).get_OC_flag() is False:
        if self.BackCell(maze).get_color() == 0:
            white_cells.append(self.BackCell(maze))
        if self.BackCell(maze).get_color() == 1:
            grey_cells.append(self.BackCell(maze))

    for cell in white_cells:
        if self.goal_cell.color == 0:
            self.WL[cell] = float('inf')
        else:
            self.WL[cell] = self.heuristic(cell, maze, key=self.key)

    return white_cells, grey_cells

def heuristic(self, cell, maze, key):
    if key == "sum_of_distance":
        return self.euclidean_dist(self.goal_cell, cell) + self.euclidean_dist(cell, self.CurrentCell(maze))
    elif key == "closest_white_to_robot":
        return self.euclidean_dist(cell, self.CurrentCell(maze))
    elif key == "closest_white_to_goal":
        return self.euclidean_dist(self.goal_cell, cell)

def Choose_direction(self, maze):
    white_cells, grey_cells = self.Cell_categorize(maze)
    
    if len(white_cells):
        next_Cell = random.choice(white_cells)
        del self.WL[next_Cell]
        print(self.color, "wl", [cell.id for cell in self.WL.keys()], self.CurrentCell(maze).id)
        return next_Cell

    if len(grey_cells) == 0:
        return self.CurrentCell(maze)

    new_grey_cells = self.sort_cell_list(grey_cells)
    next_Cell = new_grey_cells[0]
    return next_Cell

def sort_cell_list(self, cell_list):
    n = len(cell_list)
    for i in range(n):
        for j in range(n - i - 1):
            if self.visited[cell_list[j].id] > self.visited[cell_list[j + 1].id]:
                cell_list[j], cell_list[j + 1] = cell_list[j + 1], cell_list[j]

    new_list = [cell_list[0]]
    val = self.visited[cell_list[0]]
    for i in range(1, n):
        if self.visited[cell_list[i]] == val:
            new_list.append(cell_list[i])

    return new_list

def maze_exploration(self, maze, global_maze, robot_cam, lock):
    print("start")
    lock.acquire()
    self.global_maze = global_maze
    lock.release()

    while self.goal_cell.color == 0:
        lock.acquire()
        start = time.time()
        self.explore(maze, global_maze, robot_cam, lock)
        self.phase_one_counter += 1
        lock.release()

        while time.time() - start < 0.25:
            pass

    lock.acquire()
    self.AssignWall(maze, global_maze, robot_cam)
    self.Cell_categorize(maze)
    
    for cell in self.WL:
        self.WL[cell] = self.heuristic(cell, maze, key=self.key)
    
    path = find_path(maze.graph, self.CurrentCell(maze), self.goal_cell)
    lock.release()

    while not path:
        lock.acquire()
        print(self.color, "no path", [cell.id for cell in self.WL.keys()], self.CurrentCell(maze).id)
        if not len(self.WL):
            print(self.color, "empty white", self.CurrentCell(maze).id)
            self.explore(maze, global_maze, robot_cam, lock)
            path = find_path(maze.graph, self.CurrentCell(maze), self.goal_cell)
            lock.release()
            time.sleep(0.01)
            continue

        closest_white = min(self.WL, key=self.WL.get)
        if closest_white.color != 0:
            del self.WL[closest_white]
            lock.release()
            continue

        path = find_path(maze.graph, self.CurrentCell(maze), closest_white)
        lock.release()
        self.follow_path(path[0], maze, lock)
        lock.acquire()
        self.AssignWall(maze, global_maze, robot_cam)
        self.Cell_categorize(maze)
        path = find_path(maze.graph, self.CurrentCell(maze), self.goal_cell)
        lock.release()
    self.follow_path(path[0], maze, lock)
    self.CurrentCell(maze).color = 1
    self.CurrentCell(maze).OC_flag = False

def follow_path(self, path, maze, lock):
    lock.acquire()
    self.destination = path[-1]
    print(self.color, "heading to", self.destination.id)
    lock.release()

    for i in range(1, len(path)):
        lock.acquire()
        start = time.time()
        if self.destination.color == 3:
            print(self.color, "destination explored before, break!")
            lock.release()
            break

        self.next_Cell = path[i]
        self.AssignWall(maze, self.global_maze, self.robot_cam)
        self.Cell_categorize(maze)
        lock.release()

        counter = 0
        while self.next_Cell.OC_flag == True:
            time.sleep(0.1)
            lock.acquire()
            if counter > 3:
                front_robot = maze.robot_ref(self.next_Cell)
                if front_robot:
                    print(self.color, "front robot color is", front_robot.color)
                if front_robot.next_Cell == self.CurrentCell(maze):
                    lock.release()
                    maze.switch_destination(self, front_robot, lock)
                    lock.acquire()
                    print("destination switched", self.color, "heading to", self.destination.id)
            lock.release()
            lock.acquire()
            if not self.destination:
                self.explore(maze, self.global_maze, self.robot_cam)
                print(self.color, "empty dest, explore")
                lock.release()
                return

            if self.destination != path[-1]:
                print(self.color, "heading to new destination", self.destination.id)
                self.changed = "changed"
                path = find_path(maze.graph, self.CurrentCell(maze), self.destination)
                lock.release()
                self.follow_path(path[0], maze, lock)
                return

            print(self.color, "wait next cell is", self.next_Cell.id, "counter is", counter)
            lock.release()
            time.sleep(0.1)
            counter += 1

        lock.acquire()
        self.Move(maze, self.next_Cell)

        if self.CurrentCell(maze) in self.WL:
            del self.WL[self.CurrentCell(maze)]
            print(self.color, "wl", [cell.id for cell in self.WL.keys()], self.CurrentCell(maze).id)
        lock.release()

        while time.time() - start < 0.25:
            pass

    lock.acquire()
    self.AssignWall(maze, self.global_maze, self.robot_cam)
    self.Cell_categorize(maze)
    print(self.color, "follow path done, current is", self.CurrentCell(maze).id)
    self.destination = None
    lock.release()

def extract_path(self, maze, cell_a, cell_b):
    path = find_path(maze.graph, cell_a, cell_b)
    return path[0]

def add_robot_cam(self, robot_cam):
    self.robot_cam = robot_cam

def add_global_maze(self, global_maze):
    self.global_maze = global_maze
