import sys, os, time, yaml, shutil
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import cv2
import tracemalloc
# Learning CONSTANT VALUE
GAMMA = .99
MAX_STEP = int(1e3)
SEED = 3 # random seed
EPS = np.finfo(np.float32).eps.item()
MAX_DONE = 100 # condition which terminate episode
MAX_REWARD = -50 # condition which terminate learning
ALPHA = 0.05 # for the exponential moving everage
IS_CPU = not tf.test.is_gpu_available()
REWARD_DEFINITION = '''
using acrobot-v2 environment which is d2h10s edition v1.1
definition of reward : [reward = -abs(cos(theta_1))]
termination condition: [None]
'''

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

# set environment parameter
state = env.reset()
observation_n = env.observation_space.shape
action_n = env.action_space.n
hidden_n = 128
if IS_CPU:
    img_shape = env.render('rgb_array').shape[:2]

# Settings for Summary Writer of Tensorboard
# ex) Acrobot-v1_'05-14_11:04:29
start_time = time.strftime('%m-%d_%Hh-%Mm-%Ss', time.localtime())
log_dir = os.path.join(os.curdir,'logs','Acrobot-v2_'+start_time)
summary_writer = tf.summary.create_file_writer(log_dir)
os.makedirs(os.path.join(log_dir, 'video'))
shutil.copy(src=os.path.abspath(__file__),dst=os.path.join(log_dir,'source.py'))

# Learning data buffer
action_probs_buffer = []
critic_value_buffer = []
rewards_history = []
running_reward = 0
episode = 0

tracemalloc.start()
# A2C model Layer 
# load model if there is commandline argument else make new model
if len(sys.argv) > 1:
    arg_dir = sys.argv[1]
    model = tf.keras.models.load_model(os.path.join(arg_dir, 'tf_model'))
    # load progress data
    with open(os.path.join(arg_dir, 'backup.yaml')) as f:
        yaml_data = yaml.load(f)
        episode = int(yaml_data['episode']) + 1
        running_reward = int(yaml_data['running_reward'])
        episode_reward = int(yaml_data['episode_reward'])
    with open(os.path.join(log_dir, 'terminal_log.txt'), 'a') as f:
        f.write(log_dir+'\n')
        f.write('model data loaded from '+ arg_dir +'\n')
else:
    input_layer  = layers.Input(shape=observation_n)
    fc1_layer    = layers.Dense(hidden_n, activation='relu', name='Dense1')(input_layer)
    fc2_layer    = layers.Dense(hidden_n, activation='relu', name='Dense2')(fc1_layer)
    actor_layer  = layers.Dense(action_n, activation='softmax', name='Actor')(fc2_layer) # 0 <= softmax <= 1
    critic_layer = layers.Dense(1, name='Critic')(fc2_layer)

    model = keras.Model(inputs=input_layer,outputs=[actor_layer, critic_layer])

with open(os.path.join(log_dir, 'terminal_log.txt'), 'a') as f:
    f.write(REWARD_DEFINITION+'\n\n')

#keras.utils.plot_model(model, os.path.join(log_dir, "A2C_model_with_shape_info.png"), show_shapes=True)
print(model.summary())


# epsilone shouldn't too big
optimizer = optimizers.Adam(learning_rate=1e-3, epsilon=1e-3)
huber_loss = keras.losses.Huber()



while True:
    state = env.reset()
    episode_reward = 0
    done_count = 0
    deg_list = []
    with tf.GradientTape() as tape:
        # >>> save video once every 100 episode
        if IS_CPU and episode % 100 == 0:
            video_dir = os.path.join(log_dir, 'video', f'learning(episode{episode}).avi')
            videoWriter = cv2.VideoWriter(video_dir,fourcc, 15, img_shape)
        # <<< save video once every 100 episode

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
            reward = -np.abs(state[0])# state[0] - state[4]/env.MAX_VEL_1 # cos(theta_1), -1 <= cos(theta_1) <= 1
 
            rewards_history.append(reward)
            episode_reward += reward

            # Exponential Moving Everage
            # this is used to compare the end condition
            if episode == 0:
                running_reward = episode_reward
            else:
                running_reward = ALPHA * episode_reward + (1 - ALPHA) * running_reward
            deg = np.rad2deg(np.arctan2(state[1], state[0]))
            deg_list.append(deg)
            # >>> for save video
            if IS_CPU and episode % 100 == 0:
                img = env.render(mode='rgb_array').astype(np.float32)
                cv2.putText(img=img,text=f'Episode({episode:05})    Step({step:04})',\
                     org=(5,50), fontFace=font, fontScale=1,color=blue_color, thickness=1, lineType=0)
                videoWriter.write(img.astype(np.ubyte))
            # <<< for save video
            #if state[0] < -0.98:
            #    break
            
        
        if episode % 100 == 0:
            if IS_CPU:
                videoWriter.release()
            fft_list = np.abs(np.fft.fft(deg_list))
            with summary_writer.as_default():
                for step, deg in enumerate(deg_list):
                    tf.summary.scalar(f'Angle of episode{episode:05}_deg', deg, step=step)
                for step, fft in enumerate(fft_list):
                    tf.summary.scalar(f'Angle of episode{episode:05}_fft', fft, step=step)
        
        action_probs_buffer = tf.math.log(action_probs_buffer)

        # >> for monitoring
        with summary_writer.as_default():
            tf.summary.scalar('reward of episodes', episode_reward, step=episode)
        # <<< for monitoring

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
    # >>> for monitoring
    with summary_writer.as_default():
        tf.summary.scalar('losses', loss_value, step=episode)
    # <<< for monitoring

    # >>> for save model
    if episode % 100 == 0:
        model.save(os.path.join(log_dir, 'tf_model', f'learing_model{episode}'))
    # <<< for save model
    
    # >>> for backup
    now_time = time.strftime('%m-%d_%Hh-%Mm-%Ss', time.localtime())
    log_text = "running reward: {:9.2f} at episode {:8} --time:{}".format(running_reward, episode, now_time)
    print(log_text)
    with open(os.path.join(log_dir, 'terminal_log.txt'), 'a') as f:
        f.write(log_text+'\n')
    with open(os.path.join(log_dir, 'backup.yaml'), 'w') as f:
        yaml_data = {'start_time':start_time,\
                    'episode':          episode,\
                    'running_reward':   float(running_reward),\
                    'episode_reward':   float(episode_reward),\
                    'end_time':         now_time,\
                    'GAMMA':            GAMMA,\
                    'MAX_STEP':         MAX_STEP,\
                    'SEED':             SEED,\
                    'EPS':              EPS,\
                    'MAX_DONE':         MAX_DONE,\
                    'MAX_REWARD':       MAX_REWARD,\
                    'ALPHA':            ALPHA}
        yaml.dump(yaml_data, f)
    # <<< for backup
    # >>> for monitoring
    with summary_writer.as_default():
            tf.summary.scalar('running reward of episodes', running_reward, step=episode)
    # <<< for monitoring
    break
    
with summary_writer.as_default(): #텐서 보드 (그래프 추가)
  tf.summary.trace_export(
      name='graph',
      step=0,
      profiler_outdir='tensorboard/iris_species_classification_model/train')
