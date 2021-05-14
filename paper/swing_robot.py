import os, time
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import cv2

# Learning CONSTANT VALUE
GAMMA = .99
MAX_STEP = int(2e3)
SEED = 3
EPS = np.finfo(np.float32).eps.item()
DONE_CONDITION = 100

# Video Variables
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
blue_color = (255, 0, 0) # BGR
font = cv2.FONT_HERSHEY_SIMPLEX


# this can make custom environment
def init_env(uniform,custom):
    env = gym.make('Acrobot-v2')
    # make every environment have uniform random seed
    if uniform:
        env.seed(SEED)
        env.action_space.seed(SEED)
        env.observation_space.seed(SEED)
    # modify constant of envrionment
    if custom:
        env.LINK_LENGTH_1 = 0.45  # [m]
        env.LINK_LENGTH_2 = 0.10  # [m]
        env.LINK_MASS_1 = 0.3  #: [kg] mass of link 1
        env.LINK_MASS_2 = 0.4  #: [kg] mass of link 2
        env.LINK_COM_POS_1 = .225  #: [m] position of the center of mass of link 1
        env.LINK_COM_POS_2 = .05  #: [m] position of the center of mass of link 2
        env.LINK_MOI = 1.
        env.MAX_VEL_1 = 4 * np.pi #: [rad/sec]
        env.MAX_VEL_2 = 58 * 2 * np.pi / 60 #: [rad/sec] 9 * np.pi
        env.AVAIL_TORQUE = [-1., 0., +1]
    return env

# Environment creation
env = init_env(custom=False, uniform=True)
state = env.reset()
observation_n = env.observation_space.shape
action_n = env.action_space.n
hidden_n = 64
img_shape = env.render('rgb_array').shape[:2]

# A2C model Layer 
input_layer  = layers.Input(shape=observation_n)
fc1_layer    = layers.Dense(hidden_n, activation='relu', name='Dense1')(input_layer)
fc2_layer    = layers.Dense(hidden_n, activation='relu', name='Dense2')(fc1_layer)
actor_layer  = layers.Dense(action_n, activation='softmax', name='Actor')(fc2_layer)
critic_layer = layers.Dense(1, activation='tanh', name='Critic')(fc2_layer)

model = keras.Model(inputs=input_layer,outputs=[actor_layer, critic_layer])
print(model.summary())

# Settings for Summary Writer of Tensorboard
# ex) Acrobot-v1_'05-14_11:04:29
now_time = time.strftime('%m-%d_%Hh-%Mm', time.localtime())
log_dir = os.path.join(os.curdir,'logs/Acrobot-v1_'+now_time)
summary_writer = tf.summary.create_file_writer(log_dir)

optimizer = optimizers.Adam(learning_rate=1e-3)
huber_loss = keras.losses.Huber()
action_probs_buffer = []
critic_value_buffer = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:
    state = env.reset()
    
    episode_reward = 0
    torque = 0
    done_counter = 0
    img = env.render(mode='rgb_array')
    with tf.GradientTape() as tape:
        video_dir = os.path.join(os.curdir,'logs','Acrobot-v1_'+now_time,f'learning(episode{episode_count}).avi')
        if episode_count % 100 == 0:
            videoWriter = cv2.VideoWriter(video_dir,fourcc, 15, img_shape)
        for step in range(1, MAX_STEP+1):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, axis=0)
            action_probs, critic_value = model(state)
            critic_value_buffer.append(critic_value[0, 0])

            action = np.random.choice(action_n, p=np.squeeze(action_probs))
            action_probs_buffer.append(tf.math.log(action_probs[0, action]))
            state, _, done, _ = env.step(action)
            radian = np.arctan2(state[1], state[0]) # angle of link1
            reward = np.abs(radian)

            rewards_history.append(reward)
            episode_reward += reward

            if episode_count % 10 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar(f'angle of link1(episode{episode_count})', np.rad2deg(radian), step=step)
            if episode_count % 100 == 0:
                img = env.render(mode='rgb_array').astype(np.float32)
                cv2.putText(img=img,text=f'Episode({episode_count:04})    Step({step:04})', org=(50,50), fontFace=font, fontScale=1,color=blue_color, thickness=1, lineType=0)
                videoWriter.write(img.astype(np.ubyte))

            if state[0] < 0.01 :
                done_counter += 1
            else:
                done_counter = 0
            if done_counter > 1000:
                break
        videoWriter.release()

        if episode_count == 0:
            running_reward = episode_reward
        else:
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        with summary_writer.as_default():
            tf.summary.scalar('reward of episode', episode_reward, step=episode_count)
            tf.summary.scalar('running reward of episode', running_reward, step=episode_count)
        
        Returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            Returns.insert(0, discounted_sum)

        Returns = np.array(Returns)
        Returns = (Returns - np.mean(Returns)) / (np.std(Returns) + EPS)
        Returns = Returns.tolist()

        history = zip(action_probs_buffer, critic_value_buffer, Returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_probs_buffer.clear()
        critic_value_buffer.clear()
        rewards_history.clear()

        with summary_writer.as_default():
            tf.summary.scalar('losses', loss_value, step=episode_count)

    print(f"running reward: {running_reward:.2f} at episode {episode_count}")

    if running_reward > (5000):  # Co.ndition to consider the task solved
        print(f"Solved at episode {episode_count}!")
        break
    episode_count += 1


state = env.reset()
video_dir = os.path.join(os.curdir,'logs','Acrobot-v1_'+now_time,f'test.avi')
videoWriter = cv2.VideoWriter(video_dir,fourcc, 15, img_shape)
for step in range(1, MAX_STEP):
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)

    action_probs, critic_value = model(state)
    critic_value_buffer.append(critic_value[0, 0])

    action = np.argmax(action_probs)
    state, _, done, _ = env.step(action)
    radian = np.arctan2(state[1], state[0]) # angle of link1
    img = env.render(mode="rgb_array")

    with summary_writer.as_default():
        tf.summary.scalar('test angle of link1', np.rad2deg(radian), step=step)
    
    img = env.render(mode='rgb_array').astype(np.float32)
    cv2.putText(img=img,text=f'TEST: Step({step:04})', org=(50,50), fontFace=font, fontScale=1,color=blue_color, thickness=1, lineType=0)
    videoWriter.write(img.astype(np.ubyte))
    