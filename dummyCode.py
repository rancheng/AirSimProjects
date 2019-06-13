import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import time
# plot the DQN location
z = 0
pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]),
       np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]),
       np.array([0, -1, z])]
pts2 = np.squeeze(pts)
x = pts2[:, 0]
y = pts2[:, 1]
z = pts2[:, 2]
fig_0 = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)


airsim_rec_file_name = "/home/ran/Documents/AirSim/2019-06-12-15-20-28/airsim_rec.txt"
df = pd.read_csv(airsim_rec_file_name, sep='\t', lineterminator='\n')
x2 = df["POS_X"].to_list()
y2 = df["POS_Y"].to_list()
z2 = df["POS_Z"].to_list()
ax.scatter3D(x2, y2, z2)
plt.show()


def even_select(N, M):
    '''
    evenly sample M elements from list N
    :param N: number of the whole list
    :param M: number of sample size
    :return: index mask of samples
    '''
    if M > N/2:
        cut = np.zeros(N, dtype=int)
        q, r = divmod(N, N-M)
        indices = [q*i + min(i, r) for i in range(N-M)]
        cut[indices] = True
    else:
        cut = np.ones(N, dtype=int)
        q, r = divmod(N, M)
        indices = [q*i + min(i, r) for i in range(M)]
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
    with open('./Shared/reward_points.txt', 'r') as f:
        for line in f:
            point_values = line.split('\t')
            first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
            second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
            road_points.append(tuple((first_point, second_point)))

    return road_points


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


import airsim

car_client = airsim.CarClient()
car_client.confirmConnection()
reward_points = init_reward_points()
fig = plt.figure(figsize=(5, 5))
plt.ion()
ax = fig.add_subplot(111)
for point in reward_points:
    ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], 'k-', lw=2)

position_key = bytes('position', encoding='utf8')
x_val_key = bytes('x_val', encoding='utf8')
y_val_key = bytes('y_val', encoding='utf8')
pd = car_client.getCarState().kinematics_estimated.position
car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
p1, = ax.plot([car_point[0]], [car_point[1]], 'bo')
plt.show()
with open("./Shared/log_car_pos.txt", 'w') as f:
    while not car_client.getCarState().collision.has_collided:
        pd = car_client.getCarState().kinematics_estimated.position
        car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
        p1.set_xdata(car_point[0])
        p1.set_ydata(car_point[1])
        p2, = ax.plot([car_point[0]], [car_point[1]], 'go')
        fig.canvas.draw()
        fig.canvas.flush_events()
        f.write("%f\t%f\t%f\t%f\n" %())