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

class Pendulum: 

	def __init__(self): 

		self.env = gym.make('Pendulum-v0')
	
	def reset(self): 

		s = self.env.reset()
		return s.reshape(-1)
	
	def step(self, action): 
		if isinstance(action, torch.Tensor): 
			action = action.item()
		ns, r, done, infos = self.env.step(action)
		return ns.reshape(-1), r, done, infos 
	def render(self): 

		self.env.render()

	def sample_action(self): 
		return np.random.uniform(-1.,1.)
		
class ReplayBuffer: 

	def __init__(self, size = 1e6): 
		
		self.storage = []
		self.max_size = size 
		self.current = 0 
	
	def add(self, xp):
		
		if len(self.storage) == self.max_size: 	
			self.storage[self.current] = xp
			self.current = (self.current +1)%self.max_size
		else: 
			self.storage.append(xp)

	def sample(self, batch_size = 64): 

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

	def __init__(self, a, b, c): 

		super().__init__()
		
		self.policy = nn.Sequential(nn.Linear(3,400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,1), nn.Tanh())
		self.policy_targ = nn.Sequential(nn.Linear(3,400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,1), nn.Tanh())
		
		self.q = nn.Sequential(nn.Linear(4,400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,1)) 
		self.q_targ = nn.Sequential(nn.Linear(4,400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,1)) 

		self.policy_targ.load_state_dict(self.policy.state_dict())
		self.q_targ.load_state_dict(self.q.state_dict())
	
		self.adam_policy = optim.Adam(self.policy.parameters(),lr =  1e-4)
		self.adam_q = optim.Adam(self.q.parameters(), lr = 1e-2)
	def forward(self, x): 
		
		pass 
	
	def select_action(self, x): 

		return self.policy(x)*2.
	
	def evaluate(self, x, actions): 
		
		return self.q(torch.cat([x, actions], 1))

	def target_eval(self, x, actions):

		return self.q_targ(torch.cat([x, actions], 1))

	def train(self, memory, iterations = 200): 

		for i in range(iterations): 

			states, actions, rewards, next_states, masks = memory.sample(batch_size = 128)

			states = torch.tensor(states).float()
			actions = torch.tensor(actions).float().reshape(-1,1)
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

def eval(env, agent, episodes = 20): 

	reward = 0
	for _ in range(episodes): 
		done = False
		s = env.reset() 
		while not done: 

			action = agent.select_action(torch.tensor(s).float().reshape(1,-1)).item()
			action = np.clip(action + np.random.normal(0., 0.1, size = (1)), -2.,2.)[0]
			s, r, done, infos = env.step(action)

			reward += r 

	return reward/float(episodes)

memory = ReplayBuffer() 
max_episodes = 100
# agent = Agent()
agent = DDPG(3,1,2.) 
env = Pendulum()

test_reward = eval(env, agent)

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
			action = agent.select_action(torch.tensor(s).float().reshape(1,-1)).item()
			action = np.clip(action + np.random.normal(0, 0.1, size = (1)), -2. ,2.)[0]


		if ep % 10 == 0 and ep > 0: 
			env.render() 
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
			agent.train(memory, 200)
			episode_num += 1 

	if ep % 1 == 0: 
		test_reward = eval(env, agent)
		print('-'*20)
		print('Test reward: {} on 20 episodes'.format(test_reward))
		print('-'*20 + '\n\n\n')

	recap.append(test_reward)

recap_smoothed = gaussian_filter1d(recap, sigma = 3)
color = ((0.8,0.3,0.3))
plt.plot(recap, alpha = 0.3, color = color)
plt.plot(recap_smoothed, label = 'Episode reward', color = color)
plt.title('DDPG: Pendulum')
plt.legend()
plt.savefig('ddpg_pendulum.png')