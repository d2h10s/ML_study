import cv2
import gym
import os
import numpy as np

episode_count = 0
env = gym.make('Acrobot-v1')
env.reset()
img = env.render('rgb_array')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
blue_color = (255, 0, 0) # BGR
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(100):
    videoWriter = cv2.VideoWriter(f'./learning(episode{episode_count}).avi',fourcc, 15, img.shape[:2])
    for i in range(1000):
        env.step(np.random.randint(2))
        img = env.render(mode='rgb_array').astype(np.float32)
        cv2.putText(img=img,text=f'Step({i:04})   Episode{episode_count}', org=(50,50), fontFace=font, fontScale=1,color=blue_color, thickness=1, lineType=0)
        videoWriter.write(img.astype(np.ubyte))
    videoWriter.release()
    episode_count += 1