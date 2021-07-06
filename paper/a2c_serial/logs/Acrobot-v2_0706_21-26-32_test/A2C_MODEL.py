import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class a2c_model(tf.keras.Model):
    def __init__(self, observation_n, hidden_n, action_n):
        super().__init__()
        self.observation_n = observation_n
        self.hidden_n = hidden_n
        self.action_n = action_n
        self.input_layer  = layers.Input(shape=self.observation_n)
        self.fc1_layer    = layers.Dense(self.hidden_n, activation='relu', name='Dense1')(self.input_layer)
        self.fc2_layer    = layers.Dense(self.hidden_n, activation='relu', name='Dense2')(self.fc1_layer)
        self.actor_layer  = layers.Dense(self.action_n, activation='softmax', name='Actor')(self.fc2_layer) # 0 <= softmax <= 1
        self.critic_layer = layers.Dense(1, name='Critic')(self.fc2_layer)

        self.nn = keras.Model(inputs=self.input_layer, outputs=[self.actor_layer, self.critic_layer])
        print(self.nn.summary())
    
    def call(self, state):
        x = tf.expand_dims(state, axis=0)
        x = tf.convert_to_tensor(x)
        return self.nn(x)