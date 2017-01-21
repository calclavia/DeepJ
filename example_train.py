"""
A simple example to run the A3C algorithm on a toy example.
"""
import gym
import tensorflow as tf
from rl import A3CAgent
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.models import Model

env_name = 'CartPole-v1'
num_actions = 2

def make_model():
    i = Input((4,))
    x = i
    x = Dense(128, activation='relu')(x)
    policy = Dense(num_actions, activation='softmax')(x)
    value = Dense(1, activation='linear')(x)
    return Model([i], [policy, value])

with tf.device('/cpu:0'):
    agent = A3CAgent(make_model, entropy_factor=0)
    agent.train(env_name)
