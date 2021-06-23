import os
import numpy as np
import gym
import tensorflow as tf
import cv2

# >>>>>>>>>>>>>>>>
original_path = os.path.join(os.curdir,'logs', 'Acrobot-v2_0622_15-19-08_1e-3')
li = os.listdir(os.path.join(original_path, 'tf_model'))
latest_dir_name=li[np.argmax([int(x[13:]) for x in li])]
original_model_path = os.path.join(original_path, 'tf_model', latest_dir_name)
# <<<<<<<<<<<<<<<<

# Learning CONSTANT VALUE
MAX_STEP = int(1e5)
SEED = 3 # random seed


# Video Variables
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
blue_color = (255, 0, 0) # BGR
font = cv2.FONT_HERSHEY_SIMPLEX


# Settings for Summary Writer of Tensorboard
# ex) Acrobot-v1_'05-14_11:04:29
summary_writer = tf.summary.create_file_writer(original_path)

model = tf.keras.models.load_model(os.path.join(original_model_path))


env = gym.make('Acrobot-v2')
env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
state = env.reset()
video_dir = os.path.join(original_path, 'test.avi')
img_shape = env.render('rgb_array').shape[:2]
videoWriter = cv2.VideoWriter(video_dir,fourcc, 15, img_shape)
for step in range(1, MAX_STEP+1):
    print(step, end='\r')
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)

    action_probs, critic_value = model(state)

    action = np.argmax(action_probs)
    state, _, done, _ = env.step(action)
    radian = np.arctan2(state[1], state[0]) # angle of link1

    with summary_writer.as_default():
        tf.summary.scalar('test angle of link1_44', np.rad2deg(radian), step=step)
    if step > 20000:
        img = env.render(mode='rgb_array').astype(np.float32)
        cv2.putText(img=img,text=f'TEST: Step({step:04})', org=(50,50), fontFace=font, fontScale=1,color=blue_color, thickness=1, lineType=0)
        videoWriter.write(img.astype(np.ubyte))
videoWriter.release()
    