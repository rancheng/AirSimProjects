import dqn.dqn_agent
import os

#batch_update_frequency = 300
batch_update_frequency = 10
max_epoch_runtime_sec = 30
per_iter_epsilon_reduction=0.003
min_epsilon = 0.1
batch_size = 32
#replay_memory_size = 2000
replay_memory_size = 50
weights_path = '../Shared/pretrain_model_weights.h5'
train_conv_layers = 'false'
airsim_path = '/home/ran/Documents/AirsimProjects/Neighborhood/windowedNH.sh'
data_dir = os.path.join(os.getcwd(), 'Share')
experiment_name = 'local_run'