import numpy as np
import pandas as pd
import tensorboard as tb
import matplotlib.pyplot as plt
import tensorflow as tf
event_dir = "./logs/Acrobot-v1_05-18_23h-11m-57s/events.out.tfevents.1621347117.ices1011.12005.5.v2"

df = pd.DataFrame(['metric', 'value'])
for e in tf.train.summary_iterator(event_dir):
    for v in e.summary.value:
        r = {'metric': v.tag, 'value':v.simple_value}
        df.append(r, ignore_index=True)
print(df)