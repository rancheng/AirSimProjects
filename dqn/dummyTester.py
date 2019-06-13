import os

print(os.getcwd())
with open('../Shared/pretrain_model_weights.h5') as f:
    print(f.encoding)

from subprocess import call
airsim_path = '/home/ran/Documents/AirsimProjects/Neighborhood/AirSimNH.sh'
call(airsim_path + ' -ResX=640 -ResY=480 -windowed &', shell=True)