import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time
import math
import airsim
from tqdm import tqdm

mpl.style.use('seaborn')


# plot the DQN location
# z = 0
# pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]),
#        np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]),
#        np.array([0, -1, z])]
# pts2 = np.squeeze(pts)
# x = pts2[:, 0]
# y = pts2[:, 1]
# z = pts2[:, 2]
# fig_0 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, z)


# airsim_rec_file_name = "/home/ran/Documents/AirSim/2019-06-12-15-20-28/airsim_rec.txt"
# df = pd.read_csv(airsim_rec_file_name, sep='\t', lineterminator='\n')
# x2 = df["POS_X"].to_list()
# y2 = df["POS_Y"].to_list()
# z2 = df["POS_Z"].to_list()
# ax.scatter3D(x2, y2, z2)
# plt.show()


def even_select(N, M):
    '''
    evenly sample M elements from list N
    :param N: number of the whole list
    :param M: number of sample size
    :return: index mask of samples
    '''
    if M > N / 2:
        cut = np.zeros(N, dtype=int)
        q, r = divmod(N, N - M)
        indices = [q * i + min(i, r) for i in range(N - M)]
        cut[indices] = True
    else:
        cut = np.ones(N, dtype=int)
        q, r = divmod(N, M)
        indices = [q * i + min(i, r) for i in range(M)]
        cut[indices] = False

    return cut


def even_select2(len_arr, num_sample):
    idx = np.round(np.linspace(0, len_arr - 1, num_sample)).astype(int)


# idx = np.round(np.linspace(0, len(x2) - 1, 6)).astype(int)
#
# x2_nd = np.array(x2, dtype=np.int)
# y2_nd = np.array(y2, dtype=np.int)
# z2_nd = np.array(z2, dtype=np.int)
# x2_selected = x2_nd[idx]
# y2_selected = y2_nd[idx]
# z2_selected = z2_nd[idx]
# p_res = zip(x2_selected, y2_selected, z2_selected)
# p_list = []
# for ele in p_res:
#     p_list.append(np.array(ele))
#
# plt.figure()


# Reads in the reward function lines
def init_reward_points():
    road_points = []
    with open('./Shared/dump_points.txt', 'r') as f:
        for line in f:
            point_values = line.split('\t')
            first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
            second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
            road_points.append(tuple((first_point, second_point)))

    return road_points


def init_road_points():
    road_points = []
    with open('./Shared/dump_road_lines.txt', 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')] + [0])
            second_point = np.array([float(p) for p in points[1].split(',')] + [0])
            road_points.append(tuple((first_point, second_point)))
    return road_points


reward_points = init_reward_points()
road_points = init_road_points()


def map_distance(car_point):
    distance = 999
    DISTANCE_DECAY_RATE = 1.2
    for line in reward_points:
        local_distance = np.linalg.norm(np.cross(line[1] - line[0], line[1] - car_point)) / np.linalg.norm(
            line[1] - line[0])
        distance = min(local_distance, distance)

    distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
    return distance_reward


def map_reward(car_point):
    distance = 999
    DISTANCE_DECAY_RATE = 1.2
    for line in reward_points:
        local_distance = 0
        length_squared = ((line[0][0] - line[1][0]) ** 2) + ((line[0][1] - line[1][1]) ** 2)
        if (length_squared != 0):
            t = max(0, min(1, np.dot(car_point - line[0], line[1] - line[0]) / length_squared))
            proj = line[0] + (t * (line[1] - line[0]))
            local_distance = np.linalg.norm(proj - car_point)

        distance = min(local_distance, distance)

    distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
    return distance_reward


# Draws the car location plot
def draw_rl_debug(car_state, road_points):
    # fig = plt.figure(figsize=(15, 15))
    print('')
    for point in road_points:
        plt.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], 'k-', lw=2)

    position_key = bytes('position', encoding='utf8')
    x_val_key = bytes('x_val', encoding='utf8')
    y_val_key = bytes('y_val', encoding='utf8')
    pd = car_state.kinematics_estimated.position
    car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
    # car_point = np.array(
    #     [car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
    plt.plot([car_point[0]], [car_point[1]], 'bo')
    plt.draw()


def preload_reward_map():
    x_dots = np.arange(-135, 135, 0.1)
    y_dots = np.arange(-135, 135, 0.1)
    reward_map_data = np.zeros((len(x_dots), len(y_dots)))
    for i in tqdm(range(len(x_dots))):
        for j in range(len(y_dots)):
            sim_car_point = [x_dots[i], y_dots[j], 0]
            reward_map_data[i, j] = map_reward(sim_car_point)
    reward_map_data.dump('reward_map_data.dat')


def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


def test_airsim():
    car_client = airsim.CarClient()
    car_client.confirmConnection()
    reward_points = init_reward_points()
    fig = plt.figure(figsize=(5, 5))
    plt.ion()
    ax = fig.add_subplot(111)
    for point in reward_points:
        ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], 'k-', lw=2)

    # x_dots = np.arange(-130, 130, 0.1)
    # y_dots = np.arange(-130, 130, 0.1)
    # reward_map_data = np.zeros((len(x_dots), len(y_dots)))
    # for i in range(len(x_dots)):
    #     for j in range(len(y_dots)):
    #         sim_car_point = [x_dots[i], y_dots[j], 0]
    #         reward_map_data[i, j] = map_reward(sim_car_point)
    #
    # ax.plot(reward_map_data)

    pd = car_client.getCarState().kinematics_estimated.position
    car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
    arrow0 = arrow_fig([0,0], 0, 8)
    p1, = ax.plot(arrow0[0], arrow0[1], 'c-')
    # p1, = ax.plot([car_point[0]], [car_point[1]], 'bo')
    plt.show()

    with open("./Shared/log_car_pos.txt", 'w') as f:
        while not car_client.simGetCollisionInfo().has_collided:
            pd = car_client.getCarState().kinematics_estimated.position
            heading = car_client.getCarState().kinematics_estimated.orientation
            xyzw = heading.to_numpy_array()
            ypr = quaternion_to_euler(xyzw[0], xyzw[1], xyzw[2], xyzw[3])
            yaw = ypr[0]
            car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
            # arrow plot of position
            arrow_i = arrow_fig([car_point[0], car_point[1]], yaw, 8)
            p1.set_xdata(arrow_i[0])
            p1.set_ydata(arrow_i[1])
            print(map_reward(car_point))
            # dot plot of position
            # p1.set_xdata(car_point[0])
            # p1.set_ydata(car_point[1])
            # p2, = ax.plot([car_point[0]], [car_point[1]], 'r.')
            fig.canvas.draw()
            fig.canvas.flush_events()
    dummy_carpoint = [14.69277668 - 3.01696157 - 0.5962258]


def calculate_distance():
    car_client = airsim.CarClient()
    car_client.confirmConnection()
    reward_points = init_reward_points()
    fig = plt.figure(figsize=(10, 5))
    plt.ion()
    ax = fig.add_subplot(121)
    for point in reward_points:
        ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], 'k-', lw=2)
    pd = car_client.getCarState().kinematics_estimated.position
    car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
    p1, = ax.plot([car_point[0]], [car_point[1]], 'b.')
    ax2 = fig.add_subplot(122)
    plt.show()
    lsize = 20
    dislist = []
    mydislist = []
    rewardlist = []
    while not car_client.simGetCollisionInfo().has_collided:
        time.sleep(1)
        pd = car_client.getCarState().kinematics_estimated.position
        car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
        p1.set_xdata(car_point[0])
        p1.set_ydata(car_point[1])
        p2, = ax.plot([car_point[0]], [car_point[1]], 'r.')
        distance = 999
        DISTANCE_DECAY_RATE = 1.2
        distance_reward = 0
        for line in reward_points:
            local_distance = 0
            length_squared = ((line[0][0] - line[1][0]) ** 2) + ((line[0][1] - line[1][1]) ** 2)
            if (length_squared != 0):
                t = max(0, min(1, np.dot(car_point - line[0], line[1] - line[0]) / length_squared))
                proj = line[0] + (t * (line[1] - line[0]))
                local_distance = np.linalg.norm(proj - car_point)
            distance = min(local_distance, distance)
            distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
        if len(dislist) > lsize:
            dislist.append(distance)
            dislist = dislist[1:]
            mydislist.append(map_distance(car_point))
            mydislist = mydislist[1:]
            rewardlist.append(distance_reward)
            rewardlist = rewardlist[1:]
        else:
            dislist.append(distance)
            mydislist.append(map_distance(car_point))
            rewardlist.append(distance_reward)
        ax2.clear()
        p3, = ax2.plot(dislist, 'C1', label='Dist')
        ax2.plot(mydislist, 'C2', label='My Dist')
        ax2.plot(rewardlist, 'C3', label='Reward')
        ax2.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # print("Distance: %f " %distance)
        # print("My Distance: %f" %map_distance(car_point))


def arrow_fig(pos, theta, scale=1):
    return [[pos[0], pos[0] + scale * np.cos(theta + 5 * np.pi / 4), pos[0] + scale * np.sqrt(2) * np.cos(theta),
             pos[0] + scale * np.cos(theta + 3 * np.pi / 4), pos[0]],
            [pos[1], pos[1] + scale * np.sin(theta + 5 * np.pi / 4), pos[1] + scale * np.sqrt(2) * np.sin(theta),
             pos[1] + scale * np.sin(theta + 3 * np.pi / 4), pos[1]]]


# preload_reward_map()
