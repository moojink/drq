import os
import sys

import imageio
import numpy as np

import utils


class VideoRecorder(object):
    def __init__(self, view, root_dir, height=256, width=256, fps=10):
        self.view = view
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.fps = fps
        if str(self.view) == 'both':
            self.frames1 = []
            self.frames3 = []
        else:
            self.frames = []

    def init(self, enabled=True):
        if str(self.view) == 'both':
            self.frames1 = []
            self.frames3 = []
        else:
            self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            if str(self.view) == 'both':
                frame1 = env.render_overwrite(overwrite_view=1, mode='rgb_array')
                self.frames1.append(frame1)
                frame3 = env.render_overwrite(overwrite_view=3, mode='rgb_array')
                self.frames3.append(frame3)
            else:
                frame = env.render(mode='rgb_array')
                self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            if str(self.view) == 'both':
                path = os.path.join(self.save_dir, file_name + '-view_1.gif')
                imageio.mimsave(path, self.frames1, fps=self.fps)
                path = os.path.join(self.save_dir, file_name + '-view_3.gif')
                imageio.mimsave(path, self.frames3, fps=self.fps)
            else:
                path = os.path.join(self.save_dir, file_name + '.gif')
                imageio.mimsave(path, self.frames, fps=self.fps)
