import numpy as np
import ast
import re
import os
import cv2
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time
import math
import airsim
import time
from map_render import quaternion_to_euler
from tqdm import tqdm


def init_road_points():
    road_points = []
    with open('./Shared/dump_road_lines.txt', 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')] + [0])
            second_point = np.array([float(p) for p in points[1].split(',')] + [0])
            road_points.append(tuple((first_point, second_point)))
    return road_points


def connect_airsim():
    # airsim related
    car_client = airsim.VehicleClient()
    car_client.confirmConnection()
    car_client.enableApiControl(True)
    # Get the current state of the vehicle
    c_client = airsim.CarClient()
    c_client.confirmConnection()
    return car_client, c_client


def get_random_pose():
    road_points = init_road_points()
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
    # default value of direction
    random_direction = (0, 0, 0)
    # Compute the direction that the vehicle will face
    # Vertical line
    if (np.isclose(random_line[0][1], random_line[1][1])):
        if random_direction_interp > 0.5:
            random_direction = (0, 0, 0)
        else:
            random_direction = (0, 0, math.pi)
    # Horizontal line
    elif np.isclose(random_line[0][0], random_line[1][0]):
        if random_direction_interp > 0.5:
            random_direction = (0, 0, math.pi / 2)
        else:
            random_direction = (0, 0, -1.0 * math.pi / 2)

    # The z coordinate is always zero
    random_start_point[2] = -0.01
    return random_start_point, random_direction


def test_car_rand_pose(vehicle_client, car_client):
    # get the fixed location to test the orientation randomness
    starting_points_fixed, _ = get_random_pose()

    i = 0
    while i < 30:
        # print('setting position')
        starting_points, starting_direction = get_random_pose()

        # set car location and orientation
        vehicle_client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(starting_points[0], starting_points[1], starting_points[2]),
                        airsim.to_quaternion(starting_direction[0], starting_direction[1],
                                             starting_direction[2])), True)

        # test the car orientation
        # print(starting_direction)
        # car_client.simSetVehiclePose(
        #     airsim.Pose(airsim.Vector3r(starting_points_fixed[0], starting_points_fixed[1], starting_points_fixed[2]),
        #                 airsim.to_quaternion(starting_direction[0], starting_direction[1],
        #                                      starting_direction[2] + 0.01)), True)
        # print('wait for momentum die out')
        car_controls = airsim.CarControls()
        car_controls.steering = 0
        car_controls.throttle = 0
        car_controls.brake = 1
        car_client.setCarControls(car_controls)
        time.sleep(4)
        i += 1


def collect_car_rand_heading(starting_point, vehicle_client, car_client, rand_num=10, logfilename='./Data/log_rec.txt'):
    '''
    Call to set the vehicle to certain point with different orientation
    Collect the segmentation map and depth map at the same time
    :param starting_point: [x, y, z]
    :param rand_num: integer describes how many times you want to sample in a single position
    :param vehicle_client: airsim client wrapper
    :param car_client: airsim car client wrapper
    :param logfilename: log txt file name
    :return: None
    '''
    # get the fixed location to test the orientation randomness
    # starting_points_fixed, _ = get_random_pose()

    for i in range(rand_num):
        # print('setting position')
        starting_points, starting_direction = get_random_pose()
        # set car location and orientation
        # car_client.simSetVehiclePose(
        #     airsim.Pose(airsim.Vector3r(starting_points[0], starting_points[1], starting_points[2]),
        #                 airsim.to_quaternion(starting_direction[0], starting_direction[1],
        #                                      starting_direction[2])), True)
        # test the car orientation
        # print(starting_direction)
        vehicle_client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(starting_point[0], starting_point[1], starting_point[2]),
                        airsim.to_quaternion(starting_direction[0], starting_direction[1],
                                             starting_direction[2] + 0.01)), True)
        # print('wait for momentum die out')
        car_controls = airsim.CarControls()
        car_controls.steering = 0
        car_controls.throttle = 0
        car_controls.brake = 1
        car_client.setCarControls(car_controls)
        time.sleep(4)
        log_data = training_data_collector(vehicle_client, car_client)
        if log_data == "BAD_DATA":
            print(log_data)
            continue
        # print(log_data)
        log_data_txt(logfilename, log_data)


def list_encode(list):
    res = ''
    for ele in list:
        res += '#{}'.format(ele)
    return res


def img2map_coding(img):
    '''
    Note that the encoded image is the 1d array, should reshape back
    into (height, width) form
    '''
    local_encoding = []
    local_img = img.reshape((-1, 3))
    for ele in local_img:
        local_encoding.append(list_encode(list(ele)))
    return local_encoding


def segrgb2segmask(seg_img):
    if len(seg_img.shape) == 3 and seg_img.shape[-1] > 3:
        seg_img = seg_img[:, :, :3]
    seg_rgb_dict = {}
    with open("./Shared/seg_rgb.txt") as f:
        for line in f:
            (key, val) = line.split('\t')
            val_list = ast.literal_eval(re.sub('\s+', '', val))
            seg_rgb_dict[list_encode(val_list)] = int(key)
    (height, width, channel) = np.shape(seg_img)
    encoded_img = img2map_coding(seg_img)
    mask_img = [seg_rgb_dict[e] for e in encoded_img]
    mask_img = np.array(mask_img).reshape((height, width))
    return mask_img


def check_dir_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def training_data_collector(vehicle_client, car_client):
    depth_dir = check_dir_exist("./Data/Depth/")
    rgb_dir = check_dir_exist("./Data/RGB/")
    seg_dir = check_dir_exist("./Data/Seg/")
    timestamp = time.time()
    responses = vehicle_client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False),
        # segmentation image in int
        airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
        # depth in perspective projection
        airsim.ImageRequest("2", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        # scene vision image in uncompressed RGBA array
    ])
    np.shape(responses[0])
    r0 = responses[0]
    r1 = responses[1]
    r2 = responses[2]
    if not (r0.height > 0 and r1.height > 0 and r2.height > 0):
        return "BAD_DATA"
    (x_m, y_m) = np.meshgrid(range(0, r1.width), range(0, r1.height))
    if len(r0.image_data_float) > 1:
        img_seg = np.array(r0.image_data_float)
        img_seg = img_seg.reshape(r0.height, r0.width)
    else:
        img_seg = np.frombuffer(r0.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_seg = img_seg.reshape(r0.height, r0.width, 4)
    if len(r1.image_data_float) > 1:
        img_depth = np.array(r1.image_data_float)
        img_depth = img_depth.reshape(r1.height, r1.width)
        # img_depth = img_depth * 200
        # img_depth[img_depth > 255] = 255
    else:
        img_depth = np.frombuffer(r1.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_depth = img_depth.reshape(r1.height, r1.width, 4)
    if len(r2.image_data_float) > 1:
        img_rgb = np.array(r2.image_data_float)
        img_rgb = img_rgb.reshape(r2.height, r2.width)
    else:
        img_rgb = np.frombuffer(r2.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img_rgb.reshape(r2.height, r2.width, 4)
    # write the depth file
    depth_fname = os.path.join(depth_dir, str(timestamp) + '.pfm')
    airsim.write_pfm(depth_fname, airsim.get_pfm_array(r1))
    # write the rgb file
    # notice that the color of airsim published is BGRA you need to convert to RGB explicitly
    # the color is still 4 channels.
    rgb_fname = os.path.join(rgb_dir, str(timestamp) + '.png')
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2RGBA)
    # write to file
    cv2.imwrite(rgb_fname, img_rgb)
    # airsim.write_file(rgb_fname, img_rgb[:, :, :3])
    # write the segmentation file
    # note that the segmentation file has 4 channels, we need to filter out that redundent channel
    img_seg = img_seg[:, :, :3]
    segmask_img = segrgb2segmask(img_seg)
    seg_fname = os.path.join(seg_dir, str(timestamp) + '.png')
    cv2.imwrite(seg_fname, segmask_img)

    pd = car_client.getCarState().kinematics_estimated.position
    heading = car_client.getCarState().kinematics_estimated.orientation
    xyzw = heading.to_numpy_array()
    ypr = quaternion_to_euler(xyzw[0], xyzw[1], xyzw[2], xyzw[3])
    # car_point = np.array([pd.x_val, pd.y_val, pd.z_val])
    log_data_str = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(timestamp, pd.x_val, pd.y_val, pd.z_val, ypr[2],
                                                                     ypr[1], ypr[0], rgb_fname, seg_fname, depth_fname)
    return log_data_str


# code snippet of file logging:
# log the file head:
# with open("./Shared/log_car_pos_{}.txt".format(time.time()), 'w') as f:
#     # write the head of the file
#     f.write("TimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tR\tP\tY\tImg_RGB\tImg_Seg\tImg_Depth\n")


def create_log_txt_with_head(time_stamp):
    '''
    create the log file and return the log file path.
    :param time_stamp:
    :return:
    '''
    log_fname = "./Data/log_rec_{}.txt".format(str(time_stamp))
    with open(log_fname, 'w') as f:
        # write the head of the file
        f.write("TimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tR\tP\tY\tImg_RGB\tImg_Seg\tImg_Depth\n")
    return log_fname


def log_data_txt(log_file, log_data):
    # "./Shared/log_car_pos_{}.txt".format(time.time())
    with open(log_file, 'a') as f:
        # write the head of the file
        # f.write("TimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tR\tP\tY\tImg_RGB\tImg_Seg\tImg_Depth\n")
        f.write(log_data)


# veh_client, car_client = connect_airsim()
# ret_data = training_data_collector(veh_client, car_client)
# print(ret_data)


def count_sample_num_in_map():
    reward_map = np.load('./Shared/reward_map.dat')
    sample_pos_count = 0
    debug_count = 0
    for rind, row in enumerate(reward_map):
        for cind, column in enumerate(row):
            print("debug count: {}".format(debug_count))
            car_pos = [rind - 135, cind - 135]
            if column > 0:
                sample_pos_count += 1
            print("sample count: {}".format(sample_pos_count))
            debug_count += 1

def collect_by_reward_map():
    # connect to airsim
    veh_client, c_client = connect_airsim()
    # creat log data:
    timestamp = time.time()
    log_filename = create_log_txt_with_head(timestamp)
    # load reward map data
    debug_count = 0
    reward_map = np.load('./Shared/reward_map.dat', allow_pickle=True)
    for rind, row in tqdm(enumerate(reward_map)):
        for cind, column in enumerate(row):

            car_pos = [rind - 135, cind - 135, 0]
            if column > 0:
                print("r_ind, c_ind: [{}, {}]".format(rind, cind))
                collect_car_rand_heading(starting_point=car_pos, vehicle_client=veh_client, car_client=c_client,
                                         rand_num=10, logfilename=log_filename)


def collect_by_reward_map_with_fname(fname, start_r=0, start_c=0):
    # connect to airsim
    veh_client, c_client = connect_airsim()
    # load reward map data
    debug_count = 0
    reward_map = np.load('./Shared/reward_map.dat', allow_pickle=True)
    for rind, row in tqdm(enumerate(reward_map)):
        for cind, column in enumerate(row):

            car_pos = [rind - 135, cind - 135, 0]
            if rind >= start_r and cind >= start_c and column > 0:
                print("r_ind, c_ind: [{}, {}]".format(rind, cind))
                collect_car_rand_heading(starting_point=car_pos, vehicle_client=veh_client, car_client=c_client,
                                         rand_num=10, logfilename=fname)


# collect_by_reward_map()
log_datafname = "./Data/log_rec_1561337060.2409513.txt"
collect_by_reward_map_with_fname(log_datafname, start_r=6, start_c=242)