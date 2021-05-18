# sudo chmod 666 /dev/ttyACM0
# cd /home/ices/Desktop/reaction_wheel_6_v2
# python final_model_test.py

# motor voltage: 18V
# loop frequency: 40Hz (dt 0.025)

import serial
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

num_inputs = 4
num_actions = 1
num_hidden = 128
gamma = 0.99
eps = np.finfo(np.float32).eps.item()

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
actor_mean = layers.Dense(num_actions, activation="sigmoid")(common)
actor_std = layers.Dense(num_actions, activation="sigmoid")(common)
critic = layers.Dense(1)(common)

model = keras.models.load_model("my_final_model_5796")

optimizer = keras.optimizers.Adam(learning_rate=0.001)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
#episode_count = 0

next_try_time = time.time()
while True:
    try:
        #ser = serial.Serial('COM12')
        ser = serial.Serial('/dev/ttyACM0')  # lsusb # dmesg | grep tty
        #ser = serial.Serial('/dev/ttyUSB0')  # lsusb # dmesg | grep tty
        ser.baudrate = 115200
        print("Connected: " + str(ser))
        break
    except:
        ser = 0
        if time.time() >= next_try_time:
            next_try_time = next_try_time + 5
            print("Connecting")

buf_tx = bytearray()
buf_tx.append(0xAA)
buf_tx.append(0x03)
buf_tx.append(0x00)
buf_tx.append(0xBB)
ser.write(buf_tx)

buf_rx = []
count_rx = 0
prev_rad0 = math.pi
prev_rad1 = math.pi
motor_control = 0

train_state = 0 # 0 = wait, 1 = feed-forward, 2 = back-propagation
rad_dot_0_count = 0
timestep = 0

motor_change = 200
step_per_second = 40
max_step = step_per_second * 40
reward_list = []

episode_count = 0

history_list = []

while True:
    while ser.in_waiting > 0:
        data_rx = ser.read()
        #print(data_rx)
        if (ser.in_waiting == 0) and (ord(data_rx) == 0xBB):
            break

    with tf.GradientTape() as tape:
        while True:
            if ser.in_waiting > 0:
                data_rx = ser.read()
                #print(data_rx)
                buf_rx.append(ord(data_rx))
            
            if len(buf_rx) > 10:
                buf_rx = buf_rx[1:]
            
            if len(buf_rx) == 10:
                if (buf_rx[0] == 0xAA) and (buf_rx[9] == 0xBB):
                    cur_pos0 = (buf_rx[1] * 2 ** 24) + (buf_rx[2] * 2 ** 16) + (buf_rx[3] * 2 ** 8) + (buf_rx[4] * 2 ** 0) - (2 ** 32 / 2)
                    cur_pos1 = (buf_rx[5] * 2 ** 24) + (buf_rx[6] * 2 ** 16) + (buf_rx[7] * 2 ** 8) + (buf_rx[8] * 2 ** 0) - (2 ** 32 / 2)
                    buf_rx = []
                    #print('#' + str(count_rx) + ' Position0: ' + str(cur_pos0) + ' Position1: ' + str(cur_pos1) + ', Time: ' + str(time.time()))
                    count_rx = count_rx + 1

                    angle0 = (float(cur_pos0) / 512 / 4 * 360) + 180
                    rad0 = angle0 / 180 * math.pi
                    cos0 = math.cos(rad0)
                    sin0 = math.sin(rad0)
                    rad_dot0 = (rad0 - prev_rad0) * step_per_second
                    prev_rad0 = rad0

                    angle1 = (float(cur_pos1) / 200 / 4 * 360) + 180
                    rad1 = angle1 / 180 * math.pi
                    rad_dot1 = (rad1 - prev_rad1) * step_per_second
                    prev_rad1 = rad1
                    
                    #print('#' + str(count_rx) + ' Angle0: ' + str(angle0) + ', Angle1: ' + str(angle1))

                    #print('#' + str(count_rx) + ' Angle0: ' + str(angle0) + ', Rad0: ' + str(rad0) + ', Cos0: ' + str(cos0) + ', Sin0: ' + str(sin0) + ', Rad_dot0: ' + str(rad_dot0) + ', Angle1: ' + str(angle1) + ', Rad1: ' + str(rad1) + ', Rad_dot1: ' + str(rad_dot1))

                    #print('#' + str(count_rx) + ' angle1: ' + str(angle1) + ', rad_dot1: ' + str(rad_dot1))


                    """
                    # reset
                    buf_tx = bytearray()
                    buf_tx.append(0xAA)
                    buf_tx.append(0x01)
                    buf_tx.append(0x00)
                    buf_tx.append(0xBB)
                    ser.write(buf_tx)
                    """
                    
                    """
                    # motor control
                    buf_tx = bytearray()
                    buf_tx.append(0xAA)
                    buf_tx.append(0x02)
                    buf_tx.append(motor_control + 100)
                    buf_tx.append(0xBB)
                    ser.write(buf_tx)
                    """

                    if train_state == 0:
                        if (abs(rad_dot0) < (2 * math.pi * 0.01)):
                            rad_dot_0_count = rad_dot_0_count + 1
                        else:
                            rad_dot_0_count = 0

                        if abs(rad_dot1) > (2 * math.pi * 0.1):
                            buf_tx = bytearray()
                            buf_tx.append(0xAA)
                            buf_tx.append(0x02)
                            buf_tx.append(100)
                            buf_tx.append(0xBB)
                            ser.write(buf_tx)

                        if rad_dot_0_count == (step_per_second * 1):
                            buf_tx = bytearray()
                            buf_tx.append(0xAA)
                            buf_tx.append(0x01)
                            buf_tx.append(0x00)
                            buf_tx.append(0xBB)
                            ser.write(buf_tx)

                        if rad_dot_0_count >= (step_per_second * 5):    
                            rad_dot_0_count = 0
                            episode_reward = 0
                            motor_control = 100
                            print("Starting #" + str(episode_count) + ", time: "+ str(time.time()))
                            train_state = 1
                    elif train_state == 1:
                        state = [cos0, sin0, rad_dot0 / 10.0, rad_dot1 / 500.0]
                        #state = [cos, sin, rad_dot / 10.0]
                        
                        if timestep >= 1:
                            action_probs_history.append(tf.math.log(((2.718 ** (-1.0 * ((action - actor_mean_value[0, 0]) ** 2.0) / (2.0 * (actor_std_value[0, 0] ** 2.0)))) / (actor_std_value[0, 0] * ((2.0 * 3.142) ** 0.5)))))
                            reward = ((math.pi / 2) - math.acos(cos0)) / (math.pi / 2)#cos0 
                            reward_for_train = reward - (rad_dot0 ** 2 * 0.004)
                            rewards_history.append(reward_for_train)
                            episode_reward += reward
                            
                            history_list.append([rad0, rad_dot0, rad_dot1])
                            """
                            if state[0] < 0.9:
                                if state[1] > 0:
                                    history_list.append([(np.pi / 2) + (reward) * (np.pi / 2) + np.pi, rad_dot0, rad_dot1])
                                else:
                                    history_list.append([-1 * (np.pi / 2) - (reward) * (np.pi / 2) + np.pi, rad_dot0, rad_dot1])
                            else:
                                history_list.append([(np.pi / 2) + (reward) * (np.pi / 2) + np.pi, rad_dot0, rad_dot1])
                            """
                        
                        state = tf.convert_to_tensor(state)
                        state = tf.expand_dims(state, 0)

                        actor_mean_value, actor_std_value, critic_value = model(state)
                        if timestep < max_step:
                            critic_value_history.append(critic_value[0, 0])
                        
                        if timestep % 1 == 0:
                            action = float(actor_mean_value[0, 0])
                            #action = float(np.random.normal(actor_mean_value[0, 0], actor_std_value[0, 0], 1))

                        motor_control = int(75.0 - (action * 150.0))
                        if motor_control >= 100:
                            motor_control = 100
                        elif motor_control <= -100:
                            motor_control = -100

                        buf_tx = bytearray()
                        buf_tx.append(0xAA)
                        buf_tx.append(0x02)
                        buf_tx.append(motor_control + 100)
                        buf_tx.append(0xBB)
                        ser.write(buf_tx)

                        timestep = timestep + 1
                        
                        if timestep > max_step:
                            timestep = 0
                            train_state = 2
                            break
                else:
                    buf_rx = buf_rx[1:]

        if train_state == 2:
            buf_tx = bytearray()
            buf_tx.append(0xAA)
            buf_tx.append(0x02)
            buf_tx.append(125)
            buf_tx.append(0xBB)
            ser.write(buf_tx)

            if episode_count == 0:
               running_reward = episode_reward
            else:
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward    
            
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss
                critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

            loss_value = 0.2 * sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            #optimizer.apply_gradients(zip(grads, model.trainable_variables))

            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()        
            reward_list.append([episode_count, episode_reward, running_reward])

            buf_tx = bytearray()
            buf_tx.append(0xAA)
            buf_tx.append(0x03)
            buf_tx.append(0x00)
            buf_tx.append(0xBB)
            ser.write(buf_tx)

            if episode_count % 1 == 0:
                template = "episode reward: {:.2f} / {}, running reward: {:.2f} / {} at episode {}" + ", time: " + str(time.time())
                print(template.format(episode_reward, max_step, running_reward, max_step, episode_count))

            if running_reward >= (max_step * 0.5):  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))

                rewards_history = []

                with open('history_list.txt', 'wb') as f:
                    pickle.dump(history_list, f)
                history_list = []
                
                break

            episode_count += 1
            
            train_state = 0

        """    
        if cur_pos0 < 0:
            motor_control = motor_control + 10
            if motor_control >= 100:
                motor_control = 100
        else:
            motor_control = motor_control - 10
            if motor_control <= -100:
                motor_control = -100
        """

        """
        buf_tx = bytearray()
        buf_tx.append(0xAA)
        buf_tx.append(0x02)
        buf_tx.append(motor_control + 100)
        buf_tx.append(0xBB)
        ser.write(buf_tx)
        """

buf_tx = bytearray()
buf_tx.append(0xAA)
buf_tx.append(0x03)
buf_tx.append(0x00)
buf_tx.append(0xBB)
ser.write(buf_tx)
    
buf_tx = bytearray()
buf_tx.append(0xAA)
buf_tx.append(0x03)
buf_tx.append(0x00)
buf_tx.append(0xBB)
ser.write(buf_tx)
