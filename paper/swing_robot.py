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
SEED = 3 # random seed
EPS = np.finfo(np.float32).eps.item()
MAX_DONE = 100 # condition which terminate episode
MAX_REWARD = 5000 # condition which terminate learning
ALPHA = 0.05 # for the exponential moving everage
# Video Variables
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
blue_color = (255, 0, 0) # BGR
font = cv2.FONT_HERSHEY_SIMPLEX


# Environment creation
env = gym.make('Acrobot-v2')
# make every environment have uniform random seed\
env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

state = env.reset()
observation_n = env.observation_space.shape
action_n = env.action_space.n
hidden_n = 128
img_shape = env.render('rgb_array').shape[:2]

# Settings for Summary Writer of Tensorboard
# ex) Acrobot-v1_'05-14_11:04:29
now_time = time.strftime('%m-%d_%Hh-%Mm', time.localtime())
log_dir = os.path.join(os.curdir,'logs/Acrobot-v1_'+now_time)
summary_writer = tf.summary.create_file_writer(log_dir)

'''
input_layer  = layers.Input(shape=observation_n)
conv1_layer  = layers.Conv2D(32, activation='relu', name='common_1')(input_layer)
conv2_layer  = layers.Conv2D(64, activation='relu', name='common_2')(conv1_layer)
conv3_layer  = layers.Conv2D(64, activation='relu', name='common_3')(conv2_layer)
actor1_layer  = layers.Dense(action_n, activation='relu', name='Actor1')(conv3_layer) # 0 <= softmax <= 1
actor2_layer  = layers.Dense(action_n, activation='softmax', name='Actor2')(actor1_layer) # 0 <= softmax <= 1
critic1_layer  = layers.Dense(1, activation='relu', name='Critic1')(conv3_layer) # 0 <= softmax <= 1
critic2_layer = layers.Dense(1, name='Critic2')(critic1_layer)
model = keras.Model(inputs=input_layer,outputs=[actor2_layer, critic2_layer])
'''
# A2C model Layer 
input_layer  = layers.Input(shape=observation_n)
fc1_layer    = layers.Dense(hidden_n, activation='relu', name='Dense1')(input_layer)
fc2_layer    = layers.Dense(hidden_n, activation='relu', name='Dense2')(fc1_layer)
actor_layer  = layers.Dense(action_n, activation='softmax', name='Actor')(fc2_layer) # 0 <= softmax <= 1
critic_layer = layers.Dense(1, name='Critic')(fc2_layer)

model = keras.Model(inputs=input_layer,outputs=[actor_layer, critic_layer])
keras.utils.plot_model(model, os.path.join(log_dir, "A2C_model_with_shape_info.png"), show_shapes=True)
print(model.summary())


# epsilone shouldn't too big
optimizer = optimizers.Adam(learning_rate=1e-3, epsilon=1e-3)
huber_loss = keras.losses.Huber()



# Learning data buffer
action_probs_buffer = []
critic_value_buffer = []
rewards_history = []
running_reward = 0
episode = 0

while True:
    state = env.reset()
    episode_reward = 0
    done_count = 0
    with tf.GradientTape() as tape:
        video_dir = os.path.join(os.curdir,'logs', 'video', 'Acrobot-v1_'+now_time,f'learning(episode{episode}).avi')
        if episode % 100 == 0:
            videoWriter = cv2.VideoWriter(video_dir,fourcc, 15, img_shape)
        for step in range(1, MAX_STEP+1):
            state = tf.convert_to_tensor([state])

            # action_probs = tf.Tensor([[0.24805994 0.44228017 0.30965984]]
            # critic_value = tf.Tensor([[-0.24254985]]
            action_probs, critic_value = model(state)
            action = np.random.choice(action_n, p=np.squeeze(action_probs))
            action_probs_buffer.append(action_probs[0, action])
            critic_value_buffer.append(critic_value[0, 0])
            
            state, _, done, _ = env.step(action)
            # tan(theta1) [rad] = arctan(sin(theta1)/cos(theta1))
            reward = state[0] # cos(theta_1), -1 <= cos(theta_1) <= 1
 
            rewards_history.append(reward)
            episode_reward += reward

            # Exponential Moving Everage
            # this is used to compare the end condition
            if episode == 0:
                running_reward = episode_reward
            else:
                running_reward = ALPHA * episode_reward + (1 - ALPHA) * running_reward

            if episode % 100 == 0:
                with summary_writer.as_default():
                    radian = np.arctan2(state[1], state[0]) # angle of link1
                    tf.summary.scalar(f'angle of link1 at episode{episode}', np.rad2deg(radian), step=step)
                    tf.summary.scalar(f'reward=cos(th1) at episode{episode}', np.rad2deg(radian), step=step)
                img = env.render(mode='rgb_array').astype(np.float32)
                cv2.putText(img=img,text=f'Episode({episode:05})    Step({step:04})',\
                     org=(5,50), fontFace=font, fontScale=1,color=blue_color, thickness=1, lineType=0)
                videoWriter.write(img.astype(np.ubyte))

            if state[0] < 0.035:
                done_count += 1
            else:
                done_count = 0

            if done_count > MAX_DONE:
                break
        if episode % 100 == 0:
            videoWriter.release()

        action_probs_buffer = tf.math.log(action_probs_buffer)

        with summary_writer.as_default():
            tf.summary.scalar('reward of episodes', episode_reward, step=episode)
        
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
        '''
        tape.gradient(loss, x) means derivative loss of input tensor x
        '''
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_probs_buffer = []
        critic_value_buffer.clear()
        rewards_history.clear()

        with summary_writer.as_default():
            tf.summary.scalar('losses', loss_value, step=episode)
        if episode % 100 == 0:
            model.save(os.path.join(log_dir, 'model', f'A2C_model_epi{episode:05}_{now_time}'))

    print(f"running reward: {running_reward:.2f} at episode {episode} --time:{time.strftime('%m-%d_%Hh-%Mm', time.localtime())}")

    if running_reward > MAX_REWARD:  # Co.ndition to consider the task solved
        print(f"Solved at episode {episode} with running reward {running_reward}")
        break
    episode += 1


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
    