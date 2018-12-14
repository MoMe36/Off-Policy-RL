import time 
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import numpy as np 
import gym 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d

from argparse import ArgumentParser 


class GymEnv: 

	def __init__(self, env_name): 

		self.env = gym.make(env_name)
	def reset(self): 

		s = self.env.reset()
		return s.reshape(-1)
	
	def step(self, action): 
		action = action.reshape(-1)
		ns, r, done, infos = self.env.step(action)
		return ns.reshape(-1), r, done, infos 
	def render(self): 

		self.env.render()

	def sample_action(self): 
		action_space_shape = self.env.action_space.shape[0]
		action = np.random.uniform(-1,1, (action_space_shape))
		return action  
		
	def get_sizes(self): 

		return self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.action_space.high
		
class ReplayBuffer: 

	def __init__(self, size = 1e6): 
		
		self.storage = []
		self.max_size = int(size) 
		self.current = 0 
	
	def add(self, xp):
		
		if len(self.storage) == self.max_size: 	
			self.storage[self.current] = xp
			self.current = (self.current +1)%self.max_size
		else: 
			self.storage.append(xp)

	def sample(self, batch_size = 64): 

		batch_size = batch_size if len(self.storage) > batch_size else len(self.storage)

		xps = random.sample(self.storage, batch_size) 
		states, actions, rewards, next_states, dones =  [], [], [], [], []

		for i in range(batch_size): 

			xp = xps[i]
			states.append(xp[0])
			actions.append(xp[1])
			rewards.append(xp[2])
			next_states.append(xp[3])
			dones.append(xp[-1])

		return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class DDPG(nn.Module): 

	def __init__(self, obs_shape, action_shape, max_ac): 

		super().__init__()
		
		self.policy = nn.Sequential(nn.Linear(obs_shape,400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,action_shape), nn.Tanh())
		self.policy_targ = nn.Sequential(nn.Linear(obs_shape,400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,action_shape), nn.Tanh())
		
		self.q = nn.Sequential(nn.Linear(obs_shape + action_shape, 400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,1)) 
		self.q_targ = nn.Sequential(nn.Linear(obs_shape + action_shape, 400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,1)) 

		self.policy_targ.load_state_dict(self.policy.state_dict())
		self.q_targ.load_state_dict(self.q.state_dict())
	
		self.adam_policy = optim.Adam(self.policy.parameters(),lr =  1e-4)
		self.adam_q = optim.Adam(self.q.parameters(), lr = 1e-2)

		self.max_ac = torch.tensor(max_ac).float().reshape(1,-1)
	def forward(self, x): 
		
		pass 
	
	def select_action(self, x): 
	
		batch_size = x.shape[0]
		return self.policy(x).reshape(batch_size,-1)*self.max_ac
	
	def evaluate(self, x, actions): 
		
		return self.q(torch.cat([x, actions], 1))

	def target_eval(self, x, actions):

		return self.q_targ(torch.cat([x, actions], 1))

	def train(self, memory, iterations = 200): 

		for i in range(iterations): 

			states, actions, rewards, next_states, masks = memory.sample(batch_size = 128)

			states = torch.tensor(states).float()
			actions = torch.tensor(actions).float()
			rewards = torch.tensor(rewards).float().reshape(-1,1)
			next_states = torch.tensor(next_states).float()
			masks = torch.tensor(masks).float().reshape(-1,1)


			target_Q = self.target_eval(next_states, self.policy_targ(next_states))
			target_Q = rewards + (target_Q*masks*0.99).detach() 

			current_Q = self.evaluate(states, actions)
			critic_loss = F.mse_loss(current_Q, target_Q) 

			self.adam_q.zero_grad() 
			critic_loss.backward() 
			self.adam_q.step() 


			policy_loss = -torch.mean(self.evaluate(states, self.select_action(states)))
			self.adam_policy.zero_grad()
			policy_loss.backward()
			self.adam_policy.step() 


			soft_update(self.q, self.q_targ)
			soft_update(self.policy, self.policy_targ)


def soft_update(model_source, model_target, tau = 0.005) :

	for p_source, p_target in zip(model_source.parameters(), model_target.parameters()): 
		p_target.data.copy_(p_target.data*(1.-tau) + p_source.data*tau)

def eval(env, agent, memory, episodes = 20): 

	reward = 0
	for _ in range(episodes): 
		done = False
		s = env.reset() 
		while not done: 

			action = agent.select_action(torch.tensor(s).float().reshape(1,-1))
			action = action.detach().numpy().reshape(-1)
			action = np.clip(action + np.random.normal(0., 0.1, size = (action.shape)), -max_action, max_action)
			ns, r, done, infos = env.step(action)

			xp = (s, action, r, ns, 0. if done else 1.)
			memory.add(xp)

			s = ns 
			reward += r 

	return reward/float(episodes)

def make_plot(recap, args):

	plt.cla() 

	recap_smoothed = gaussian_filter1d(np.array(recap), sigma = 8)
	color = ((0.8,0.3,0.3))
	plt.plot(recap, alpha = 0.3, color = color)
	plt.plot(recap_smoothed, label = 'Episode reward', color = color)
	plt.title('DDPG: {}'.format(args.env_name))
	plt.legend()
	plt.savefig('./runs/ddpg_{}.png'.format(args.env_name))

def save_model(agent, args): 

	with open('./trained_models/{}_model.txt'.format(args.env_name), 'w') as file: 
		file.write('{}'.format(agent))
	torch.save(agent.policy.state_dict(), './trained_models/{}_params'.format(args.env_name))

def get_args(): 

	parser = ArgumentParser()
	parser.add_argument('--env_name', default = 'Pendulum-v0')
	parser.add_argument('--nb_episodes', type = int, default = 800)


	return parser.parse_args()  


args = get_args()

memory = ReplayBuffer() 
max_episodes = args.nb_episodes

env = GymEnv(args.env_name)

obs_size, ac_size, max_action  = env.get_sizes()
agent = DDPG(obs_size, ac_size, max_action) 

test_reward = eval(env, agent, memory)
print('-'*20)
print('Initial eval: {}'.format(test_reward))
print('-'*20)

total_timesteps = 0 
episode_num = 0 
recap = []

s = env.reset()
for ep in range(max_episodes): 
	s = env.reset() 
	done = False 
	episode_reward = 0 
	while not done: 

		if ep < 20: 
			action = env.sample_action()
		else: 
			action = agent.select_action(torch.tensor(s).float().reshape(1,-1)).detach().numpy().reshape(-1)
			action = np.clip(action + np.random.normal(0, 0.1, size = (action.shape)), -max_action , max_action)
			
		if ep % 10 == 0 and ep > 0: 
			env.render() 
			if args.env_name == "Pendulum-v0":
				time.sleep(0.02)

		ns, r, done, infos = env.step(action)
		xp = (s, action, r, ns, 0. if done else 1.)
		memory.add(xp)

		s = ns 
		episode_reward += r 

		total_timesteps += 1
		if done: 
			if ep == 20: 
				print('Starting to use agent to select action \n\n')
			print('Episode {}\nTotal timesteps: {}\nEpisode reward: {}\n'.format(episode_num, total_timesteps, episode_reward))
			agent.train(memory, 50)
			episode_num += 1 

	if ep % 1 == 0: 
		test_reward = eval(env, agent, memory)
		print('-'*20)
		print('Test reward: {} on 20 episodes'.format(test_reward))
		print('-'*20 + '\n\n\n')

	recap.append(test_reward)
	make_plot(recap, args)
	save_model(agent, args)

