import io, os, yaml, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from pytz import timezone, utc
from datetime import datetime as dt


STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

class a2c_agent():
    def __init__(self, model, lr=1e-3, sampling_time=0.025, suffix=""):
        self.model = model
        self.GAMMA = .99
        self.MAX_STEP = 1000
        self.EPS = np.finfo(np.float32).eps.item()
        self.ALPHA = 0.01
        self.LEARNING_RATE = lr
        self.EPSILON = 1e-3
        self.MAX_DONE = 20

        self.num_episode = 1
        self.episode_reward = 0
        self.EMA_reward = 0
        self.SUFFIX = suffix
        self.sampling_time = sampling_time

        self.start_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
        self.start_time_str = dt.strftime(self.start_time, '%m%d_%H-%M-%S')
        self.log_dir = os.path.join(os.curdir,'logs','Acrobot-v2_'+self.start_time_str+self.SUFFIX)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.optimizer = optimizers.Adam(learning_rate=self.LEARNING_RATE, epsilon=self.EPSILON)
        self.huber_loss = keras.losses.Huber()
        

    def init_message(self, msg):
        with open(os.path.join(self.log_dir, 'terminal_log.txt'), 'a') as f:
            f.write(msg+'\n\n')
    def fft(self, deg_list):
        Fs = 1/self.sampling_time
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
        x = np.arange(n)*self.sampling_time
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

    def train(self, env):
        done_cnt = 0
        while env.ser.isOpen():
            try:
                state = env.reset()
                self.episode_reward = 0
                discounted_sum = 0
                deg_list = []
                action_probs_buffer = []
                critic_value_buffer = []
                rewards_history = []
                Returns = []
                actor_losses = []
                critic_losses = []
                with tf.GradientTape(persistent=False) as tape:
                    for step in range(1, self.MAX_STEP+1):
                        start_time = time.time()
                        state = tf.convert_to_tensor(state)
                        #print(state)
                        action_probs, critic_value = self.model(state)
                        action = np.random.choice(self.model.action_n, p=np.squeeze(action_probs))
                        action_probs_buffer.append(action_probs[0, action])
                        critic_value_buffer.append(critic_value[0, 0])
                        state = env.step(action)
                        reward = -np.abs(state[0])

                        rewards_history.append(reward)
                        self.episode_reward += reward

                        if self.num_episode == 0:
                            self.EMA_reward = self.episode_reward
                        else:
                            self.EMA_reward = self.ALPHA * self.episode_reward + (1 - self.ALPHA) * self.EMA_reward
                        deg = np.rad2deg(np.arctan2(state[1], state[0]))
                        deg_list.append(deg)

                        didWait = False
                        while time.time() - start_time < self.sampling_time:
                            didWait = True
                        if not didWait:
                            print("never wait")

                    action_probs_buffer = tf.math.log(action_probs_buffer)

                    for r in rewards_history[::-1]:
                        discounted_sum = r + self.GAMMA * discounted_sum
                        Returns.insert(0, discounted_sum)

                    Returns = np.array(Returns)
                    Returns = (Returns - np.mean(Returns)) / (np.std(Returns) + self.EPS)
                    Returns = Returns.tolist()

                    history = zip(action_probs_buffer, critic_value_buffer, Returns)
                    for log_prob, value, Return in history:
                        advantage = Return - value
                        actor_losses.append(-log_prob * advantage)
                        critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(Return, 0)))

                    loss_value = sum(actor_losses) + sum(critic_losses)

                    grads = tape.gradient(loss_value, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                most_freq, sigma, plot_img = self.fft(deg_list)

                if self.num_episode % 100 == 0 or self.num_episode == 1:
                    self.model.save(os.path.join(self.log_dir, 'tf_model', f'learing_model{self.num_episode}'))
                    with self.summary_writer.as_default():
                        tf.summary.image(f'fft of episode{self.num_episode:05}', plot_img, step=0)

                del deg_list
                del tape, grads
                del actor_losses, critic_losses
                del action_probs_buffer, critic_value_buffer
                del rewards_history, Returns

                # >>> for monitoring
                with self.summary_writer.as_default():
                    tf.summary.scalar('losses', loss_value, step=self.num_episode)
                    tf.summary.scalar('reward of episodes', self.episode_reward, step=self.num_episode)
                    tf.summary.scalar('frequency of episodes', most_freq, step=self.num_episode)
                    tf.summary.scalar('sigma of episodes', sigma, step=self.num_episode)
                # <<< for monitoring

                now_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
                now_time_str = dt.strftime(now_time, '%m-%d_%Hh-%Mm-%Ss')
                log_text = "EMA reward: {:9.2f} at episode {:5} --freq:{:7.3f} --sigma:{:7.2f} --time:{} ".format(self.EMA_reward, self.num_episode, most_freq, sigma, now_time_str)
                print(log_text)
                with open(os.path.join(self.log_dir, 'terminal_log.txt'), 'a') as f:
                    f.write(log_text+'\n')

                self.yaml_backup()

                if 100 < sigma < 200 and 0.3 < most_freq < 0.55:
                    done_cnt += 1
                else:
                    done_cnt = 0
                if done_cnt > self.MAX_DONE:
                    print(f"Solved at episode {self.num_episode} with EMA reward {self.EMA_reward}")
                    with self.summary_writer.as_default():
                        tf.summary.image(f'fft of final episode{self.num_episode:05}', plot_img, step=0)
                    break
                self.num_episode += 1
            except Exception as e:
                print(e, 'error occured in train loop')
                time.sleep(1000)


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

    def yaml_backup(self):
        now_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
        now_time_str = dt.strftime(now_time, '%m-%d_%Hh-%Mm-%Ss')
        with open(os.path.join(self.log_dir, 'backup.yaml'), 'w') as f:
            yaml_data = {'START_TIME':      self.start_time_str,\
                        'ELAPSED_TIME':     str(now_time-self.start_time),\
                        'END_TIME':         now_time_str,\
                        'GAMMA':            self.GAMMA,\
                        'MAX_STEP':         self.MAX_STEP,\
                        'EPS':              self.EPS,\
                        'ALPHA':            self.ALPHA,\
                        'LEARNING_RATE':    self.LEARNING_RATE,\
                        'EPSILON':          self.EPSILON,\
                        'EPISODE':          self.num_episode,\
                        'EMA_REWARD':       float(self.EMA_reward),\
                        'self.EPISODE_REWARD':   float(self.episode_reward)}
            yaml.dump(yaml_data, f)