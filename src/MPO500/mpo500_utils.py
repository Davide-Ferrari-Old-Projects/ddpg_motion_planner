#!usr/bin/env/python3

import numpy as np
import math


class MPO500():

    """ Mobile Industrial Robots Neobotix MPO500 utilities.

    Attributes:
    
        max_lin_vel (float): Maximum robot's linear velocity (m/s).
        min_lin_vel (float): Minimum robot's linear velocity (m/s).
        max_ang_vel (float): Maximum robot's angular velocity (rad/s).
        min_ang_vel (float): Minimum robot's linear velocity (rad/s).

    ROS Command (cmd_vel):
    
        geometry_msgs/Twist: 
            
            linear:
                x: 0.0
                y: 0.0
                z: 0.0
            
            angular:
                x: 0.0
                y: 0.0
                z: 0.0"

    """

    def __init__(self):

        # Keep Conservative (max ~? 1.0)
        self.max_lin_vel = 0.5
        self.min_lin_vel = -0.5

        # Keep Conservative (max ~? 1.5)
        self.max_ang_vel = 0.7
        self.min_ang_vel = -0.7

    def get_max_lin_vel(self):

        return self.max_lin_vel

    def get_min_lin_vel(self):

        return self.min_lin_vel

    def get_max_ang_vel(self):

        return self.max_ang_vel

    def get_min_ang_vel(self):

        return self.min_ang_vel

    def get_corners_positions(self, x, y, yaw):
        
        """ Get robot's corners coordinates given the coordinates of its center.
        
        https://www.neobotix-robots.com/products/mobile-robots/mobile-robot-mpo-500

        Args:
        
            x (float): x coordinate of robot's geometric center.
            y (float): y coordinate of robot's geometric center.
            yaw (float): yaw angle of robot's geometric center.

        The coordinates are given with respect to the map origin and cartesian system.

        Returns:
        
            list[list]: x and y coordinates of the 4 robot's corners.

        """

        robot_x_dimension = 0.986
        robot_y_dimension = 0.662
        dx = robot_x_dimension/2
        dy = robot_y_dimension/2

        delta_corners = [[dx, -dy], [dx, dy], [-dx, dy], [-dx, -dy]]
        corners = []

        for corner_xy in delta_corners:

            # Rotate point around origin
            r_xy = rotate_point(corner_xy[0], corner_xy[1], yaw)

            # Translate back from origin to corner
            corners.append([sum(x) for x in zip(r_xy, [x, y])])

        return corners


def rotate_point(x, y, theta):

    """ Rotate a point around the origin by an angle theta.
    
    Args:
    
        x (float): x coordinate of point.
        y (float): y coordinate of point.
        theta (float): rotation angle (rad).
        
    Returns:
    
        list: [x,y] coordinates of rotated point.
        
    """

    x_r = x * math.cos(theta) - y * math.sin(theta)
    y_r = x * math.sin(theta) + y * math.cos(theta)

    return [x_r, y_r]
