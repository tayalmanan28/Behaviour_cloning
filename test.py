from __future__ import print_function

import os
import gym
import numpy as np
import torch

from train import data_transform, actions_set, Net, DATA_DIR, MODEL_FILE
from pyglet.window import key

def play(model):
    """
    Let the agent play
    :param model: the network
    """
    env = gym.make('CarRacing-v0')

    # use ESC to exit
    global exit_test
    exit_test = False

    def key_press(k, mod):
        global exit_test
        if k == key.ESCAPE: exit_test = True

    # initialize environment
    state = env.reset()
    env.unwrapped.viewer.window.on_key_press = key_press

    while 1:
        env.render()

        state = np.moveaxis(state, 2, 0)  # channel first image

        # numpy to tensor
        state = torch.from_numpy(np.flip(state, axis=0).copy())
        state = data_transform(state)  # apply transformations
        state = state.unsqueeze(0)  # add additional dimension

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(state)

        normalized = torch.nn.functional.softmax(outputs, dim=1)

        # translate from net output to env action
        max_action = np.argmax(normalized.cpu().numpy()[0])
        action = actions_set[max_action]

        # adjust brake power
        if action[2] != 0:
            action[2] = 0.3

        state, _, terminal, _ = env.step(action)  # one step

        # if terminal:
        #     env.close()
        #     return

        if exit_test:
            env.close()
            return


if __name__ == '__main__':
    m = Net()
    m.load_state_dict(torch.load(os.path.join(DATA_DIR, MODEL_FILE)))
    m.eval()
    play(m)
