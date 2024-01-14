import cv2 as cv
import numpy as np
import time

class Detector:
    def __init__(self, maze, name, lock, robots=[]):
        self.robots = robots
        self.maze = maze
        self.name = name
        self.robot_count = 0
        self.thickness = 2
        self.w = 40
        self.lock = lock

        for i in robots:
            if i:
                self.robot_count += 1
            self.initialize()

    def initialize(self):
        start = time.time()
        w = 40
        maze = self.maze
        Row = len(self.maze.cell)
        Col = len(self.maze.cell[0])
        self.img = np.zeros([(Row + 1) * w, (Col + 3) * w, 3], dtype=np.uint8)
        self.img.fill(255)
        
        for i in range(Row):
            for j in range(Col):
                if self.name == "global" and 0 < i < Row and 0 < j < Col:
                    font = cv.FONT_HERSHEY_SIMPLEX
                    xx = int(w * (j + 0.1))
                    yy = int(w * (i + 0.5))
                    cv.putText(self.img, str(maze.cell[i][j].id), (xx, yy), font, 0.4, (0, 0, 0), self.thickness // 10, cv.LINE_AA)

                if maze.cell[i][j].downWall == 0:
                    x1 = w * j
                    y1 = w * (i + 1)
                    x2 = w * (j + 1)
                    y2 = w * (i + 1)
                    cv.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), self.thickness, lineType=8)

                if maze.cell[i][j].rightWall == 0:
                    x1 = w * (j + 1)
                    y1 = w * i
                    x2 = w * (j + 1)
                    y2 = w * (i + 1)
                    cv.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), self.thickness, lineType=8)

        for robot in self.robots:
            self.set_color(robot.CurrentCell(self.maze), robot.color, robot)
        
        self.present()

    def present(self):
        if self.name == "global":
            cv.imshow(self.name, self.img)
            cv.moveWindow(self.name, 1000, 60)
            cv.waitKey(1)
        else:
            cv.moveWindow(self.name, 100, 60)
            cv.imshow(self.name, self.img)
            cv.waitKey(1)

    def renew_image(self, key, last_cell, next_cell):
        pass

    def add_right_wall(self, cell):
        i = cell.Row
        j = cell.Col
        w = self.w
        x1 = w * (j + 1)
        y1 = w * i
        x2 = w * (j + 1)
        y2 = w * (i + 1)
        cv.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), self.thickness, lineType=8)

    def add_left_wall(self, cell):
        i = cell.Row
        j = cell.Col
        w = self.w
        x1 = w * j
        y1 = w * i
        x2 = w * j
        y2 = w * (i + 1)
        cv.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), self.thickness, lineType=8)

    def add_up_wall(self, cell):
        i = cell.Row
        j = cell.Col
        w = self.w
        x1 = w * j
        y1 = w * (i + 1)
        x2 = w * (j + 1)
        y2 = w * (i + 1)
        cv.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), self.thickness, lineType=8)

    def add_down_wall(self, cell):
        i = cell.Row
        j = cell.Col
        w = self.w
        x1 = w * j
        y1 = w * (i + 1)
        x2 = w * (j + 1)
        y2 = w * (i + 1)
        cv.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), self.thickness, lineType=8)

    def set_color(self, cell, color, robot=None):
        i = cell.Row
        j = cell.Col
        w = self.w

        if robot is None:
            if color == 1:
                rgb_color = (160, 160, 160)
            elif color == 2:
                rgb_color = (0, 0, 0)

            start_point = (w * j + self.thickness, w * i + self.thickness)
            end_point = (w * (j + 1) - self.thickness, w * (i + 1) - self.thickness)
            cv.rectangle(self.img, start_point, end_point, rgb_color, -1)
        else:
            color = robot.rgb
            dir_r = robot.direction
            r = cell.Row
            c = cell.Col
            center = (int(w * c + w / 2), int(w * r + w / 2))

            if r:
                cv.circle(self.img, center, int(w / 3), color, self.thickness)

            if dir_r == 0:
                rr, cc = -1, 0
            elif dir_r == 1:
                rr, cc = 0, -1
            elif dir_r == 2:
                rr, cc = 1, 0
            else:
                rr, cc = 0, 1

            cent = (int(center[0] + rr * w / 7), int(center[1] + cc * w / 7))
            cv.circle(self.img, cent, int(w / 8), color, self.thickness)

        self.present()

    def visualize(self):
        start = time.time()
        w = self.w
        thickness = 2
        maze = self.maze
        self.img = np.zeros([(Row + self.robot_count + 1) * w, (Col + 3) * w, 3], dtype=np.uint8)
        self.img.fill(255)

        if maze:
            Row = len(maze.cell)
            Col = len(maze.cell[0])
            img = np.zeros([(Row + self.robot_count + 1) * w, (Col + 3) * w, 3], dtype=np.uint8)
            img.fill(255)

            for i in range(Row):
                for j in range(Col):
                    if maze.cell[i][j].downWall == 0:
                        x1 = w * j
                        y1 = w * (i + 1)
                        x2 = w * (j + 1)
                        y2 = w * (i + 1)
                        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness, lineType=8)

                    if maze.cell[i][j].rightWall == 0:
                        x1 = w * (j + 1)
                        y1 = w * i
                        x2 = w * (j + 1)
                        y2 = w * (i + 1)
                        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness, lineType=8)

                    if maze.cell[i][j].color == 1:
                        start_point = (w * j + thickness, w * i + thickness)
                        end_point = (w * (j + 1) - thickness, w * (i + 1) - thickness)
                        cv.rectangle(img, start_point, end_point, (160, 160, 160), -1)

                    if maze.cell[i][j].color == 2:
                        start_point = (w * j + thickness, w * i + thickness)
                        end_point = (w * (j + 1) - thickness, w * (i + 1) - thickness)
                        cv.rectangle(img, start_point, end_point, (0, 0, 0), -1)

            count = 0

            for robot in self.robots:
                if robot:
                    color = robot.rgb
                    dir_r = robot.direction
                    txt = f"{robot.color}: current cell: {robot.RowInd} {robot.ColInd} "
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img, txt, (w, (Row + 1 + count) * w), font, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                    r = robot.CurrentCell(maze).Row
                    c = robot.CurrentCell(maze).Col
                    center = (int(w * c + w / 2), int(w * r + w / 2))

                    if r:
                        cv.circle(img, center, int(w / 3), color, thickness)

                    if dir_r == 0:
                        rr, cc = -1, 0
                    elif dir_r == 1:
                        rr, cc = 0, -1
                    elif dir_r == 2:
                        rr, cc = 1, 0
                    else:
                        rr, cc = 0, 1

                    cent = (int(center[0] + rr * w / 7), int(center[1] + cc * w / 7))
                    cv.circle(img, cent, int(w / 8), color, thickness)
                count += 1

            if self.name == "global":
                cv.imshow(self.name, img)
                cv.moveWindow(self.name, 1000, 60)
                cv.waitKey(1)
            else:
                cv.moveWindow(self.name, 100, 60)
                cv.imshow(self.name, img)
                cv.waitKey(1)
