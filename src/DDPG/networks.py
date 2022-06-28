#!usr/bin/env/python3

import os
import tensorflow as tf
from keras import Model 
from keras.layers import Dense

class CriticNetwork(Model):

    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir='tmp/ddpg'):

        # SuperConstructor
        super(CriticNetwork, self).__init__()

        # Save Variables
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # Model Variables
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        # Create Network (2 Fully Connected Layers and a Final Output Layer)
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):

        # Pass the Concatenated State and Action throug the FC1 and Concatenate Along the 1st Axis
        action_value = self.fc1(tf.concat([state, action], axis=1))

        # Pass the Output to the 2nd Fully-Connected Layer
        action_value = self.fc2(action_value)

        # Pass to the Output Layer
        q = self.q(action_value)

        return q


class ActorNetwork(Model):

    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512, name='actor', chkpt_dir='tmp/ddpg'):

        # SuperConstructor
        super(ActorNetwork, self).__init__()

        # Save Variables
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Model Variables
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        # Create Network (2 Fully Connected Layers and a Final Output Layer)
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state, bounds=1):

        # Pass the State to the 1st and the 2nd Fully-Connected Layer
        prob = self.fc1(state)
        prob = self.fc2(prob)

        # Pass to the Output Layer (multiply by the bounds if they are not +/- 1)
        mu = self.mu(prob) * bounds

        return mu
