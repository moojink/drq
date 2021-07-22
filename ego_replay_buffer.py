import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.img_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.ee_grip_obses = np.empty((capacity, 3), dtype=np.float32)
        self.ee_pos_rel_base_obses = np.empty((capacity, 9), dtype=np.float32)
        self.contact_flags_obses = np.empty((capacity, 6), dtype=np.float32)

        self.next_img_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_ee_grip_obses = np.empty((capacity, 3), dtype=np.float32)
        self.next_ee_pos_rel_base_obses = np.empty((capacity, 9), dtype=np.float32)
        self.next_contact_flags_obses = np.empty((capacity, 6), dtype=np.float32)

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        img_obs, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs = obs
        np.copyto(self.img_obses[self.idx], img_obs)
        np.copyto(self.ee_grip_obses[self.idx], ee_grip_obs)
        np.copyto(self.ee_pos_rel_base_obses[self.idx], ee_pos_rel_base_obs)
        np.copyto(self.contact_flags_obses[self.idx], contact_flags_obs)

        next_img_obs, next_ee_grip_obs, next_ee_pos_rel_base_obs, next_contact_flags_obs = next_obs
        np.copyto(self.next_img_obses[self.idx], next_img_obs)
        np.copyto(self.next_ee_grip_obses[self.idx], next_ee_grip_obs)
        np.copyto(self.next_ee_pos_rel_base_obses[self.idx], next_ee_pos_rel_base_obs)
        np.copyto(self.next_contact_flags_obses[self.idx], next_contact_flags_obs)

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        img_obses = self.img_obses[idxs]
        next_img_obses = self.next_img_obses[idxs]
        img_obses_aug = img_obses.copy()
        next_img_obses_aug = next_img_obses.copy()

        img_obses = torch.as_tensor(img_obses, device=self.device).float()
        next_img_obses = torch.as_tensor(next_img_obses, device=self.device).float()
        img_obses_aug = torch.as_tensor(img_obses_aug, device=self.device).float()
        next_img_obses_aug = torch.as_tensor(next_img_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        img_obses = self.aug_trans(img_obses)
        next_img_obses = self.aug_trans(next_img_obses)

        img_obses_aug = self.aug_trans(img_obses_aug)
        next_img_obses_aug = self.aug_trans(next_img_obses_aug)

        ee_grip_obses = torch.as_tensor(self.ee_grip_obses[idxs], device=self.device).float()
        ee_pos_rel_base_obses = torch.as_tensor(self.ee_pos_rel_base_obses[idxs], device=self.device).float()
        contact_flags_obses = torch.as_tensor(self.contact_flags_obses[idxs], device=self.device).float()
        next_ee_grip_obses = torch.as_tensor(self.next_ee_grip_obses[idxs], device=self.device).float()
        next_ee_pos_rel_base_obses = torch.as_tensor(self.next_ee_pos_rel_base_obses[idxs], device=self.device).float()
        next_contact_flags_obses = torch.as_tensor(self.next_contact_flags_obses[idxs], device=self.device).float()

        obses = (img_obses, ee_grip_obses, ee_pos_rel_base_obses, contact_flags_obses)
        next_obses = (next_img_obses, next_ee_grip_obses, next_ee_pos_rel_base_obses, next_contact_flags_obses)
        obses_aug = (img_obses_aug, ee_grip_obses, ee_pos_rel_base_obses, contact_flags_obses)
        next_obses_aug = (next_img_obses_aug, next_ee_grip_obses, next_ee_pos_rel_base_obses, next_contact_flags_obses)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
