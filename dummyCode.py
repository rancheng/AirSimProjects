import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
# plot the DQN location
z = 0
pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]),
       np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]),
       np.array([0, -1, z])]
pts2 = np.squeeze(pts)
x = pts2[:, 0]
y = pts2[:, 1]
z = pts2[:, 2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)


airsim_rec_file_name = "/home/ran/Documents/AirSim/2019-06-08-19-53-07/airsim_rec.txt"
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

idx = np.round(np.linspace(0, len(x2) - 1, 6)).astype(int)

x2_nd = np.array(x2, dtype=np.int)
y2_nd = np.array(y2, dtype=np.int)
z2_nd = np.array(z2, dtype=np.int)
x2_selected = x2_nd[idx]
y2_selected = y2_nd[idx]
z2_selected = z2_nd[idx]
p_res = zip(x2_selected, y2_selected, z2_selected)
p_list = []
for ele in p_res:
    p_list.append(np.array(ele))

