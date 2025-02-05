import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import ego_utils as utils
import hydra


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, view, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_logits = False
        self.feature_dim = feature_dim
        if str(view) == 'both':
            # If using both views 1 and 3, use half the hidden dimensions for
            # view 1 and the other half for view 3.
            output_dim = feature_dim // 2
        else:
            output_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        if obs_shape[1] == 84: # DeepMind control suite images are 84x84
            conv_out_size = 35
        elif obs_shape[1] == 128:
            conv_out_size = 57
        else:
            raise ValueError("Unsupported image size.")

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * conv_out_size * conv_out_size, output_dim),
            nn.LayerNorm(output_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, view, encoder_cfg, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds, proprio_obs_shape):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.view = view

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim + proprio_obs_shape, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        if str(self.view) == 'both':
            img_obs1, img_obs3, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs = obs
            encoder_out1 = self.encoder(img_obs1, detach=detach_encoder)
            encoder_out3 = self.encoder(img_obs3, detach=detach_encoder)
            obs_out = torch.cat((encoder_out1, encoder_out3, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs), dim=-1)
        else:
            img_obs, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs = obs
            encoder_out = self.encoder(img_obs, detach=detach_encoder)
            obs_out = torch.cat((encoder_out, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs), dim=-1)

        mu, log_std = self.trunk(obs_out).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, view, encoder_cfg, action_shape, hidden_dim, hidden_depth, proprio_obs_shape):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.view = view

        self.Q1 = utils.mlp(self.encoder.feature_dim + proprio_obs_shape + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + proprio_obs_shape + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        if str(self.view) == 'both':
            img_obs1, img_obs3, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs = obs
            assert img_obs1.size(0) == action.size(0)
            assert img_obs3.size(0) == action.size(0)
            encoder_out1 = self.encoder(img_obs1, detach=detach_encoder)
            encoder_out3 = self.encoder(img_obs3, detach=detach_encoder)
            obs_out = torch.cat((encoder_out1, encoder_out3, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs), dim=-1)
        else:
            img_obs, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs = obs
            assert img_obs.size(0) == action.size(0)
            encoder_out = self.encoder(img_obs, detach=detach_encoder)
            obs_out = torch.cat((encoder_out, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs), dim=-1)

        obs_action = torch.cat([obs_out, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, view, obs_shape, proprio_obs_shape, action_shape, action_range, device,
                 encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.view = view

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        if str(self.view) == 'both':
            img_obs1, img_obs3, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs = obs
            img_obs1 = torch.FloatTensor(img_obs1).to(self.device)
            img_obs1 = img_obs1.unsqueeze(0)
            img_obs3 = torch.FloatTensor(img_obs3).to(self.device)
            img_obs3 = img_obs3.unsqueeze(0)
        else:
            img_obs, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs = obs
            img_obs = torch.FloatTensor(img_obs).to(self.device)
            img_obs = img_obs.unsqueeze(0)
        ee_grip_obs = torch.FloatTensor(ee_grip_obs).to(self.device)
        ee_grip_obs = ee_grip_obs.unsqueeze(0)
        ee_pos_rel_base_obs = torch.FloatTensor(ee_pos_rel_base_obs).to(self.device)
        ee_pos_rel_base_obs = ee_pos_rel_base_obs.unsqueeze(0)
        contact_flags_obs = torch.FloatTensor(contact_flags_obs).to(self.device)
        contact_flags_obs = contact_flags_obs.unsqueeze(0)
        if str(self.view) == 'both':
            obs = img_obs1, img_obs3, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs
        else:
            obs = img_obs, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def save_checkpoint(self, log_dir, step):
        torch.save(
            {
                'step': step,
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict(),
            },
            os.path.join(log_dir, str(step) + '.ckpt')
        )

    def load_checkpoint(self, checkpoint_dir, checkpoint_step):
        checkpoint_path = checkpoint_dir + '/' + str(checkpoint_step) + '.ckpt'
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_state_dict'])