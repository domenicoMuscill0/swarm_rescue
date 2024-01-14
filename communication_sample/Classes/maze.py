import random
from datetime import datetime
import os
from copy import copy
from dijkstar import Graph, find_path


class Cell:
    def __init__(self, Row, Col, id):
        self.Row = Row
        self.Col = Col
        self.id = id
        self.xpos = 0
        self.ypos = 0
        self.downWall = 1  # 1 for way
        self.rightWall = 1  # 0 for wall
        self.color = 0
        self.par = self
        self.OC_flag = False
        self.graph = Graph()

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def set_OC_flag(self, flag):
        self.OC_flag = flag

    def get_OC_flag(self):
        return self.OC_flag


class Maze:
    def __init__(self, Row, Col, StartRow=0, StartCol=0, FinishRow=6, FinishCol=1):
        self.StartRow = StartRow
        self.StartCol = StartCol
        self.FinishRow = FinishRow
        self.FinishCol = FinishCol
        self.Row = Row + 1
        self.Col = Col + 1
        self.cell = [[Cell(row, col, (self.Col - 1) * (row - 1) + col) for col in range(self.Col)] for row in
                     range(self.Row)]
        self.connected_components = (len(self.cell) - 1) * (len(self.cell[0]) - 1)
        self.graph = Graph()

    def add_robots(self, robots):
        self.robots = robots

    def get_id_from_cell(self, cell):
        return cell.id

    def get_cell_from_id(self, id):
        row = id // (self.Col - 1) + 1
        col = id - (self.Col - 1) * (row - 1) - col
        return self.cell[row][col]

    def up_cell(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if row_u == 0:
            return None
        return self.cell[row_u - 1][col_u]

    def up_wall(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if row_u == 0:
            return None
        return self.up_cell(cell_u).downWall

    def down_cell(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if row_u == self.Row - 1:
            return None
        return self.cell[row_u + 1][col_u]

    def down_wall(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if row_u == self.Row - 1:
            return None
        return cell_u.downWall

    def left_cell(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if col_u == 0:
            return None
        return self.cell[row_u][col_u - 1]

    def left_wall(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if col_u == 0:
            return None
        return self.left_cell(cell_u).rightWall

    def right_cell(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if col_u == self.Col - 1:
            return None
        return self.cell[row_u][col_u + 1]

    def right_wall(self, cell_u):
        row_u, col_u = cell_u.Row, cell_u.Col
        if col_u == self.Col - 1:
            return None
        return cell_u.rightWall

    def set_obstacle(self, cell_u, cell_v, robot_cam, val):
        if not cell_u or not cell_v:
            return
        row_u, col_u = cell_u.Row, cell_u.Col
        row_v, col_v = cell_v.Row, cell_v.Col
        if row_u == row_v:
            if col_u < col_v:
                cell_u.rightWall = val
                if not val:
                    robot_cam.add_right_wall(cell_u)
                else:
                    cell_v.rightWall = val
                    if not val:
                        robot_cam.add_right_wall(cell_v)
        else:
            if row_u < row_v:
                cell_u.downWall = val
                if not val:
                    robot_cam.add_down_wall(cell_u)
            else:
                cell_v.downWall = val
                if not val:
                    robot_cam.add_down_wall(cell_v)


    def add_obstacle(self, cell_u, cell_v, robot_cam=None):
        self.set_obstacle(cell_u, cell_v, robot_cam, 0)

    def remove_obstacle(self, cell_u, cell_v, robot_cam=None):
        self.set_obstacle(cell_u, cell_v, robot_cam, 1)

    def add_to_graph(self, cell_u, cell_v):
        self.graph.add_edge(cell_u, cell_v, 1)
        self.graph.add_edge(cell_v, cell_u, 1)

    def build(self):
        Row = len(self.cell)
        Col = len(self.cell[0])
        self.cell[Row - 1][Col - 1].xpos = -2.25
        self.cell[Row - 1][Col - 1].ypos = -2.25
        self.cell[1][1].xpos = 2.25
        self.cell[1][1].ypos = 2.25
        self.cell[Row - 1][1].xpos = -2.25
        self.cell[Row - 1][1].ypos = 2.25
        self.cell[1][Col - 1].xpos = 2.25
        self.cell[1][Col - 1].ypos = 2.25

        for r in range(1, Row):
            for c in range(1, Col):
                self.cell[r][c].xpos = 2.25 - 0.5 * (r - 1)
                self.cell[r][c].ypos = 2.25 - 0.5 * (c - 1)

        for i in range(1, Row):
            self.cell[i][Col - 1].rightWall = 0
            self.cell[i][0].rightWall = 0

        for i in range(1, Col):
            self.cell[0][i].downWall = 0
            self.cell[Row - 1][i].downWall = 0

    def find(self, u):
        if u != u.par:
            u.par = self.find(u.par)
        return u.par

    def union(self, cell_u, cell_v):
        if not cell_u or cell_u.Row == 0 or cell_u.Col == 0 or cell_u.Row == self.Row or cell_u.Col == self.Col:
            return False
        if not cell_v or cell_v.Row == 0 or cell_v.Col == 0 or cell_v.Row == self.Row or cell_v.Col == self.Col:
            return False

        par_u = self.find(cell_u)
        par_v = self.find(cell_v)

        if par_u == par_v:
            return False

        par_u.par = par_v
        self.remove_obstacle(cell_u, cell_v)  # Remove obstacle between the cells
        self.connected_components -= 1

        return True

    def build_maze(self):
        self.build()
        self.make_all_obstacles()
        self.generate_random_obstacles()

    def make_all_obstacles(self):
        for i in range(1, self.Row):
            for j in range(1, self.Col):
                self.cell[i][j].rightWall = 0
                self.cell[i][j].downWall = 0

    def generate_random_cell(self):
        i = random.randint(1, self.Row - 1)
        j = random.randint(1, self.Col - 1)
        return self.cell[i][j]

    def return_cell(self, pos):
        return self.cell[pos[0]][pos[1]]

    def generate_random_obstacles(self):
        random.seed(datetime.now())
        while self.connected_components > 1:
            cur_cell = self.generate_random_cell()
            lst = [self.up_cell(cur_cell), self.down_cell(cur_cell), self.left_cell(cur_cell), self.right_cell(cur_cell)]
            random_cell = random.choice(lst)

            if self.union(cur_cell, random_cell):
                pass

    def build_graph(self, global_maze):
        for i in range(1, self.Row):
            for j in range(1, self.Col):
                curr_cell = self.cell[i][j]
                global_cell = global_maze.cell[i][j]

                if self.right_cell(global_cell) and self.right_wall(global_cell):
                    self.add_to_graph(curr_cell, self.right_cell(curr_cell))

                if self.down_cell(global_cell) and self.down_wall(global_cell):
                    self.add_to_graph(curr_cell, self.down_cell(curr_cell))

    def get_robots_pos(self):
        return [self.robots[i].CurrentCell(self) for i in range(len(self.robots))]

    def robot_ref(self, cell):
        for robot in self.robots:
            if robot.CurrentCell(self) == cell:
                return robot
        return None

    def switch_destination(self, robot1, robot2, lock):
        print("here")
        lock.acquire()

        if robot1.color > robot2.color:
            lock.release()
            return

        robot1.destination, robot2.destination = robot2.destination, robot1.destination
        print("in switch")
        lock.release()
