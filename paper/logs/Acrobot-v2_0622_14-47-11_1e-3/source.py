import sys, io, os, yaml, shutil
import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from pytz import timezone, utc
from datetime import datetime as dt

# Learning CONSTANT VALUE
GAMMA = .99
MAX_STEP = int(1e3)
SEED = 3 # random seed
EPS = np.finfo(np.float32).eps.item()
MAX_DONE = 100 # condition which terminate episode
MAX_REWARD = -400 # condition which terminate learning
ALPHA = 0.05 # for the exponential moving everage
LEARNING_RATE = 1e-3
EPSILON = 1e-3
INIT_MESSAGE = '''
using acrobot-v2 environment which is d2h10s edition v2.0
definition of reward : [reward = -abs(cos(theta_1))]
termination condition: [None]
'''
SUFFIX = "_1e-3"


def fft(deg_list):
    Ts = 0.025
    Fs = 1/Ts
    n = len(deg_list)
    scale = n//2
    k = np.arange(n)
    T = n/Fs
    freq = k/T
    freq = freq[range(scale)]
    fft_data = np.fft.fft(deg_list)/n
    fft_data = fft_data[range(scale)]
    fft_mag_data = np.abs(fft_data)
    most_freq = freq[np.argmax(fft_mag_data)]
    x = np.arange(n)*Ts
    sigma = np.max(fft_mag_data)/np.mean(fft_mag_data)
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.title('FFT')
    plt.plot(x, deg_list)
    plt.xlabel('sec')
    plt.ylabel('deg')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.grid(True)
    plt.ylabel('mag')
    plt.xlabel('frequency')
    plt.plot(freq, fft_mag_data, linestyle=' ', marker='^', linewidth=1)
    plt.vlines(freq, [0], fft_mag_data)
    plt.xlim([0, 4])
    plt.legend([f'most freq:{most_freq:2.3f}Hz', f'sigma: {sigma:5.2f}'])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_image = tf.image.decode_png(buf.getvalue(), channels=4)
    plot_image = tf.expand_dims(plot_image, 0)
    return most_freq, sigma, plot_image


def yaml_backup(episode, EMA_reward, episode_reward, now_time, now_time_str):
    with open(os.path.join(log_dir, 'backup.yaml'), 'w') as f:
        yaml_data = {'START_TIME':      start_time_str,\
                    'ELAPSED_TIME':     str(now_time-start_time),\
                    'END_TIME':         now_time_str,\
                    'GAMMA':            GAMMA,\
                    'MAX_STEP':         MAX_STEP,\
                    'SEED':             SEED,\
                    'EPS':              EPS,\
                    'MAX_DONE':         MAX_DONE,\
                    'MAX_REWARD':       MAX_REWARD,\
                    'ALPHA':            ALPHA,\
                    'LEARNING_RATE':    LEARNING_RATE,\
                    'EPSILON':          EPSILON,\
                    'EPISODE':          episode,\
                    'EMA_REWARD':       float(EMA_reward),\
                    'EPISODE_REWARD':   float(episode_reward)}
        yaml.dump(yaml_data, f)


def run_test():
    state = env.reset()
    for step in range(1, MAX_STEP):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_probs, critic_value = model(state)

        action = np.argmax(action_probs)
        state, _, done, _ = env.step(action)
        radian = np.arctan2(state[1], state[0]) # angle of link1

        with summary_writer.as_default():
            tf.summary.scalar('test angle of link1', np.rad2deg(radian), step=step)


# Settings for Summary Writer of Tensorboard
# ex) Acrobot-v1_'05-14_11:04:29
start_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
start_time_str = dt.strftime(start_time, '%m%d_%H-%M-%S')
log_dir = os.path.join(os.curdir,'logs','Acrobot-v2_'+start_time_str+SUFFIX)
summary_writer = tf.summary.create_file_writer(log_dir)
shutil.copy(src=os.path.abspath(__file__), dst=os.path.join(log_dir,'source.py'))

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

# A2C model Layer 
# load model if there is commandline argument else make new model
if len(sys.argv) > 1:
    arg_dir = sys.argv[1]
    model = tf.keras.models.load_model(os.path.join(arg_dir, 'tf_model'))
    # load progress data
    with open(os.path.join(arg_dir, 'backup.yaml')) as f:
        yaml_data = yaml.load(f)
        episode = int(yaml_data['episode']) + 1
        EMA_reward = int(yaml_data['EMA_reward'])
        episode_reward = int(yaml_data['episode_reward'])
        GAMMA = GAMMA
        MAX_STEP = MAX_STEP
        SEED = SEED
        EPS = EPS
        MAX_DONE = MAX_DONE
        MAX_REWARD = MAX_REWARD
        ALPHA = ALPHA
        LEARNING_RATE = LEARNING_RATE
        EPSILON = EPSILON

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
print(model.summary())
#keras.utils.plot_model(model, os.path.join(log_dir, "A2C_model_with_shape_info.png"), show_shapes=True)

optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON)
huber_loss = keras.losses.Huber()

with open(os.path.join(log_dir, 'terminal_log.txt'), 'a') as f:
    f.write(INIT_MESSAGE+'\n\n')



episode = 0
while True:
    state = env.reset()
    episode_reward = 0
    done_count = 0
    discounted_sum = 0
    deg_list = []
    action_probs_buffer = []
    critic_value_buffer = []
    rewards_history = []
    Returns = []

    with tf.GradientTape(persistent=False) as tape:
        for step in range(1, MAX_STEP+1):
            state = tf.convert_to_tensor([state])

            # action_probs = tf.Tensor([[0.24805994 0.44228017 0.30965984]]
            # critic_value = tf.Tensor([[-0.24254985]]
            action_probs, critic_value = model(state)
            action = np.random.choice(action_n, p=np.squeeze(action_probs))
            action_probs_buffer.append(action_probs[0, action])
            critic_value_buffer.append(critic_value[0, 0])
            
            state, _, done, _ = env.step(action)
            reward = -np.abs(state[0])
 
            rewards_history.append(reward)
            episode_reward += reward

            if episode == 0:
                EMA_reward = episode_reward
            else:
                EMA_reward = ALPHA * episode_reward + (1 - ALPHA) * EMA_reward
            deg = np.rad2deg(np.arctan2(state[1], state[0]))
            deg_list.append(deg)
            
        action_probs_buffer = tf.math.log(action_probs_buffer)
        
        for r in rewards_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            Returns.insert(0, discounted_sum)

        Returns = np.array(Returns)
        Returns = (Returns - np.mean(Returns)) / (np.std(Returns) + EPS)
        Returns = Returns.tolist()

        history = zip(action_probs_buffer, critic_value_buffer, Returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, Return in history:
            advantage = Return - value
            actor_losses.append(-log_prob * advantage)
            critic_losses.append(tf.stop_gradient(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(advantage, 0))))

        loss_value = sum(actor_losses) + sum(critic_losses)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    

    most_freq, sigma, plot_img = fft(deg_list)

    if episode % 100 == 0:
        model.save(os.path.join(log_dir, 'tf_model', f'learing_model{episode}'))
        with summary_writer.as_default():
            tf.summary.image(f'fft of episode{episode:05}', plot_img, step=0)
    
    del deg_list
    del tape, grads
    del actor_losses, critic_losses
    del action_probs_buffer, critic_value_buffer
    del rewards_history, Returns

    # >>> for monitoring
    with summary_writer.as_default():
        tf.summary.scalar('losses', loss_value, step=episode)
        tf.summary.scalar('reward of episodes', episode_reward, step=episode)
        tf.summary.scalar('EMA reward of episodes', EMA_reward, step=episode)
        tf.summary.scalar('frequency of episodes', most_freq, step=episode)
        tf.summary.scalar('sigma of episodes', sigma, step=episode)
    # <<< for monitoring
    
    now_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
    now_time_str = dt.strftime(now_time, '%m-%d_%Hh-%Mm-%Ss')
    log_text = "EMA reward: {:9.2f} at episode {:5} --freq:{:7.3f} --sigma:{:7.2f} --time:{} ".format(EMA_reward, episode, most_freq, sigma, now_time_str)
    print(log_text)
    with open(os.path.join(log_dir, 'terminal_log.txt'), 'a') as f:
        f.write(log_text+'\n')
    
    yaml_backup(episode, EMA_reward, episode_reward, now_time, now_time_str)
    
    if EMA_reward > MAX_REWARD:
        print(f"Solved at episode {episode} with EMA reward {EMA_reward}")
        with summary_writer.as_default():
            tf.summary.image(f'fft of final episode{episode:05}', plot_img, step=0)
        break

    episode += 1


run_test()