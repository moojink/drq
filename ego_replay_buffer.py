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

        self.obses = [0] * capacity # list of placeholders for observations
        self.next_obses = [0] * capacity
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self.obses[self.idx] = obs
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        self.next_obses[self.idx] = next_obs
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = list(map(self.obses.__getitem__, idxs))
        next_obses = list(map(self.next_obses.__getitem__, idxs))

        # Extract the image observations from the batch.
        img_obses = [obs['im_rgb'] for obs in obses]
        img_obses_aug = img_obses.copy()
        next_img_obses = [obs['im_rgb'] for obs in next_obses]
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

        # Turn the list of obs dicts into a dict of concatenated tensors.
        ee_grip_obses = torch.as_tensor([obs['ee_grip'] for obs in obses], device=self.device).float()
        ee_pos_rel_base_obses = torch.as_tensor([obs['ee_pos_rel_base'] for obs in obses], device=self.device).float()
        contact_flags_obses = torch.as_tensor([obs['contact_flags'] for obs in obses], device=self.device).float()
        next_ee_grip_obses = torch.as_tensor([obs['ee_grip'] for obs in next_obses], device=self.device).float()
        next_ee_pos_rel_base_obses = torch.as_tensor([obs['ee_pos_rel_base'] for obs in next_obses], device=self.device).float()
        next_contact_flags_obses = torch.as_tensor([obs['contact_flags'] for obs in next_obses], device=self.device).float()
        obses = dict(
            im_rgb = img_obses,
            ee_grip = ee_grip_obses,
            ee_pos_rel_base = ee_pos_rel_base_obses,
            contact_flags = contact_flags_obses
        )
        obses_aug = dict(
            im_rgb = img_obses_aug,
            ee_grip = ee_grip_obses,
            ee_pos_rel_base = ee_pos_rel_base_obses,
            contact_flags = contact_flags_obses
        )
        next_obses = dict(
            im_rgb = next_img_obses,
            ee_grip = next_ee_grip_obses,
            ee_pos_rel_base = next_ee_pos_rel_base_obses,
            contact_flags = next_contact_flags_obses
        )
        next_obses_aug = dict(
            im_rgb = next_img_obses_aug,
            ee_grip = next_ee_grip_obses,
            ee_pos_rel_base = next_ee_pos_rel_base_obses,
            contact_flags = next_contact_flags_obses
        )

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
