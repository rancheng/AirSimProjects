import numpy as np
import pandas as pd
temp = []
rwd_filename = './reward_points.txt'
dump_filename = './dump_points.txt'
df = pd.read_csv(rwd_filename, header=None, sep='\t', lineterminator='\n')
df = df + 125
