from __future__ import print_function

import gzip
import os
import pickle
import time
import argparse
from pyglet.window import key

import gym
import numpy as np

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'

def key_press(k, mod):
    global agent_action, restart_train, exit_train, pause_train, acceleration
    
    if k == key.ENTER: restart_train = True
    if k == key.ESCAPE: exit_train = True
    if k == key.SPACE:pause_train = not pause_train

    if k == key.UP:
        acceleration = True
        agent_action[1] = 1.0
        agent_action[2] = 0

    if k == key.DOWN: 
        agent_action[2] = 1

    if k == key.LEFT:
        agent_action[0] = -1.0
        agent_action[1] = 0.0   # no acceleration while turning

    if k == key.RIGHT:
        agent_action[0] = +1.0
        agent_action[1] = 0.0   # no acceleration when turning

def key_release(k, mod):
    global agent_action, acceleration
    if k == key.UP: 
        acceleration = False
        agent_action[1] = 0.0

    if k == key.DOWN:
        agent_action[2] = 0.0

    if k == key.LEFT:
        agent_action[0] = 0
        agent_action[1] = acceleration  # restore acceleration

    if k == key.RIGHT:
        agent_action[0] = 0
        agent_action[1] = acceleration  # restore acceleration

def rollout(env):
    global restart_train, agent_action, exit_train, pause_train
    ACTIONS = env.action_space.shape[0]
    agent_action = np.zeros(ACTIONS, dtype=np.float32)
    exit_train = False
    pause_train = False
    restart_train = False

    # if the file exists, append
    if os.path.exists(os.path.join(DATA_DIR, DATA_FILE)):
        with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
            observations = pickle.load(f)
    else:
        observations = list()

    state = env.reset()
    total_reward = 0
    episode = 1
    while 1:
        env.render()
        a = np.copy(agent_action)
        old_state = state
        if agent_action[2] != 0:
            agent_action[2] = 0.1

        state, reward, done, info = env.step(agent_action)

        observations.append((old_state, a, state, reward, done))

        total_reward += reward

        if exit_train:
            env.close()
            return

        if restart_train:
            restart_train = False
            state = env.reset()
            continue

        if done:
                
            if episode == 20:
                # store generated data
                data_file_path = os.path.join(DATA_DIR, DATA_FILE)
                print("Saving observations to " + data_file_path)

                if not os.path.exists(DATA_DIR):
                    os.mkdir(DATA_DIR)

                with gzip.open(data_file_path, 'wb') as f:
                    pickle.dump(observations, f)
                
                env.close()
                return

            print("Episodes %i reward %0.2f" % (episode, total_reward))

            episode += 1

            state = env.reset()

        while pause_train:
            env.render()
            time.sleep(0.1)


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(env.action_space.shape[0]))

    rollout(env)
