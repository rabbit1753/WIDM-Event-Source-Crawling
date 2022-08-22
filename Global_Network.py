"""
https://jovian.ai/tobiasschmidbauer1312/asynchronous-actor-to-critic-ac3
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

class ShareAdam(torch.optim.Adam):
    def __init__(self, params, lr=0.001,  betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(ShareAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) #explain what is going on here

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_links, gamma):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.actor_layer1 = nn.Linear(input_dims, 128)
        self.actor_layer = nn.Linear(128, n_links)
        
        self.critic_layer1 = nn.Linear(input_dims, 128)
        self.crtitc_layer = nn.Linear(128, n_links)   

        self.rewards = []
        self.links = []
        self.state = [[]for i in range(10000)]
    
    def forward(self, state):
        actor_layer1 = F.relu(self.actor_layer1(state))
        probability = torch.sigmoid(self.actor_layer(actor_layer1)) #dont discuss with other link,so use sigmoid
        
        critic_layer1 = F.relu(self.critic_layer1(state)) 
        scores = self.actor_layer(critic_layer1)
        
        return probability, scores
    
    def record_episode(self, reward, link, state_count):
        self.rewards.append(reward)
        self.links.append(link)
        self.state[state_count].append(reward,link) 

    def clear_memory(self):
        self.rewards = []
        self.links = []
        self.state = [[]for i in range(10000)]

    def calc_R(self, done):   #done from event estimator conclus
        state = torch.tensor(self.states, dtype = torch.float)
        _,  value= self.forward(state)

        award = value[-1]*(1-int(done))  #dont know how to handle ,maybe game characteristic

        reward_record = []
        for reward in self.rewards[::-1]:
            award = reward + self.gamma*award
            reward_record.append(award)
        
        return reward_record
    
    def calc_loss(self, done): #cal loss function
        state = torch.tensor(self.states, dtype=torch.float)
        #actions = torch.tensor(self.actions, dtype=torch.float)

        reward = self.calc_R(done)

        _ , critic = self.forward(state)
        values = critic.squeeze()
        critic_loss = (reward - values) 
        """
        probs = torch.sigmod(actor, dim = 1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(reward - values)
        """
        total_loss = critic_loss
        return total_loss

    # def choose_action(prob): #frontier solve?
        
    #     action = np.random.choice(a = 2, p = prob[0].detach().numpy())
    #     return action

if __name__=="__main__":

    print("HI")
