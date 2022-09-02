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
    def __init__(self, input_dims, n_links, gamma = 0.9):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.actor_layer1 = nn.Linear(*input_dims, 128)
        self.actor_layer = nn.Linear(128, 1)
        
        self.critic_layer1 = nn.Linear(*input_dims, 128)
        self.crtitc_layer = nn.Linear(128, 1)   

        self.rewards = []
        self.actions = []
        self.state = []
        
    def forward(self, state):
        actor_layer1 = F.relu(self.actor_layer1(state))
        probability = torch.sigmoid(self.actor_layer(actor_layer1)) #dont discuss with other link,so use sigmoid

        critic_layer1 = F.relu(self.critic_layer1(state))
        scores = torch.sigmoid(self.crtitc_layer(critic_layer1))
        
        return probability, scores
    
    def record_episode(self, reward, action, feature):
        self.rewards.append(reward)
        self.actions.append(action)
        self.state.append(feature)

    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.state = []

    def calc_R(self, state):   #done from event estimator #要改
        print("執行過reward")
        # state = torch.from_numpy(np.asarray(self.state))
        _,  value= self.forward(state)

        award = value[-1]  #dont know how to handle ,maybe game characteristic

        reward_record = []
        for reward in self.rewards[::-1]:
            award = reward + self.gamma*award
            reward_record.append(award)
        
        return reward_record
    
    def calc_loss(self):

        actions = torch.tensor(self.actions)

        temp = []
        ma = 0
        for i in range(len(self.state[0][0])):
            temp.append(0)
        temp = torch.tensor(temp)
        temp = torch.unsqueeze(temp,0)
        for i in range(len(self.state)):
            max_temp = len(self.state[i])
            if ma < max_temp:
                ma = max_temp
                max_index = i
        for i in range(len(self.state)):
            if i == max_index:
                continue
            for j in range(len(self.state[i]),len(self.state[max_index])):
                self.state[i] = torch.cat((self.state[i],temp),0)
        state = torch.tensor([item.detach().numpy() for item in self.state])
        
        # actions = torch.tensor(self.actions, dtype=torch.float)

        acc = self.calc_R(state)
        acc = torch.tensor([item.detach().numpy() for item in acc])
        actor , critic = self.forward(state)
        predict = critic.squeeze() 
        acc = torch.reshape(acc, (acc.shape[0],acc.shape[1]))
        acc_use = []
        predict_use = []
        for i in range(len(actions)):
            acc_use.append(acc[i][actions[i]])
            predict_use.append(predict[i][actions[i]])
        acc_use = torch.tensor(acc_use)
        predict_use = torch.tensor(predict_use)
        critic_loss = (acc_use - predict_use)**2
        # print(critic_loss)

        probs = torch.sigmoid(actor)
        probs = probs.squeeze()
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        # acc = acc[0,2:4]
        # predict = predict[0,2:4]
        actor_loss = -log_probs*(acc_use - predict_use)

        total_loss = (critic_loss + actor_loss).mean()
        print(actor_loss,critic_loss,total_loss)
        return total_loss

if __name__=="__main__":
    lr = 1e-4
    #example
    state = [[[0.25,0.67],[0.44,0.11]],[[0.43,0.69],[0.25,0.67],[0.44,0.11]]]
    input_dims = 2
    n_links = 1
    global_actor_critic = ActorCritic(input_dims ,n_links)
