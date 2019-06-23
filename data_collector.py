import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time
import math
import airsim


def init_road_points():
    road_points = []
    with open('./Shared/dump_road_lines.txt', 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')] + [0])
            second_point = np.array([float(p) for p in points[1].split(',')] + [0])
            road_points.append(tuple((first_point, second_point)))
    return road_points

road_points = init_road_points()

# airsim related
car_client = airsim.VehicleClient()
car_client.confirmConnection()
car_client.enableApiControl(True)
car_controls = airsim.CarControls()
# Get the current state of the vehicle
car_state = car_client.getCarState()

def get_random_pose():
    # Pick a random road.
    random_line_index = np.random.randint(0, high=len(road_points))

    # Pick a random position on the road.
    # Do not start too close to either end, as the car may crash during the initial run.
    random_interp = (np.random.random_sample() * 0.4) + 0.3

    # Pick a random direction to face
    random_direction_interp = np.random.random_sample()

    # Compute the starting point of the car
    random_line = road_points[random_line_index]
    random_start_point = list(random_line[0])
    random_start_point[0] += (random_line[1][0] - random_line[0][0]) * random_interp
    random_start_point[1] += (random_line[1][1] - random_line[0][1]) * random_interp

    # Compute the direction that the vehicle will face
    # Vertical line
    if (np.isclose(random_line[0][1], random_line[1][1])):
        if (random_direction_interp > 0.5):
            random_direction = (0, 0, 0)
        else:
            random_direction = (0, 0, math.pi)
    # Horizontal line
    elif (np.isclose(random_line[0][0], random_line[1][0])):
        if (random_direction_interp > 0.5):
            random_direction = (0, 0, math.pi / 2)
        else:
            random_direction = (0, 0, -1.0 * math.pi / 2)

    # The z coordinate is always zero
    random_start_point[2] = 0.01
    return (random_start_point, random_direction)

starting_points, starting_direction = get_random_pose()