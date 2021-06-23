import sys, io, os, yaml, shutil
import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from pytz import timezone, utc
from datetime import datetime as dt

class a2c_agent():
    def __init__(self, model, lr=1e-3, gamma=0.99, alpha=0.05, max_step=1000, summary_writer=None):
        self.GAMMA = gamma
        self.MAX_STEP = max_step
        self.EPS = np.finfo(np.float32).eps.item()
        self.ALPHA = alpha
        self.LEARNING_RATE = lr
        self.EPSILON = 1e-3

        self.summary_writer = summary_writer
        self.model = model

    def train(self, env):
        while True:
            state = env.reset()
            episode_reward = 0
            discounted_sum = 0
            deg_list = []
            action_probs_buffer = []
            critic_value_buffer = []
            rewards_history = []
            Returns = []

            with tf.GradientTape(persistent=False) as tape:
                for step in range(1, self.MAX_STEP+1):
                    state = tf.expand_dims(state, axis=0)
                    state = tf.convert_to_tensor(state)

                    action_probs, critic_value = self.model(state)
                    action = np.random.choice(self.model.action_n, p=np.squeeze(action_probs))
                    action_probs_buffer.append(action_probs[0, action])
                    critic_value_buffer.append(critic_value[0, 0])
                       
                    new_state, reward, done, info = env.step(action)
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
                for log_prob, value, ret in history:
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)
                    critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

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
            
            if 100 < sigma < 200 and 0.3 < most_freq < 0.5:
                print(f"Solved at episode {episode} with EMA reward {EMA_reward}")
                with summary_writer.as_default():
                    tf.summary.image(f'fft of final episode{episode:05}', plot_img, step=0)
                break

            episode += 1

    def run_test(self, env):
        state = env.reset()
        for step in range(1, self.MAX_STEP):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, _ = self.model(state)

            action = np.argmax(action_probs)
            state, *_ = env.step(action)
            radian = np.arctan2(state[1], state[0]) # angle of link1

            with self.summary_writer.as_default():
                tf.summary.scalar('test angle of link1', np.rad2deg(radian), step=step)

    def yaml_backup(self, episode, EMA_reward, episode_reward, now_time, now_time_str):
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