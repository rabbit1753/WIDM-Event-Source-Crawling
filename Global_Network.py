"""
https://jovian.ai/tobiasschmidbauer1312/asynchronous-actor-to-critic-ac3
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        self.links = []
        self.state = []
        
    def forward(self, state):
        print("執行過forward")
        # print(state.shape)
        actor_layer1 = F.relu(self.actor_layer1(state))
        # probability = self.actor_layer(actor_layer1)
        probability = torch.sigmoid(self.actor_layer(actor_layer1)) #dont discuss with other link,so use sigmoid
        print("執行過這裡")
        # critic_layer1 = F.relu(self.critic_layer1(state[count]))
        critic_layer1 = F.relu(self.critic_layer1(state))
        # score = self.critic_layer(critic_layer1)
        scores = torch.sigmoid(self.crtitc_layer(critic_layer1))
        
        return probability, scores
    
    def record_episode(self, reward, link, feature):
        self.rewards.append(reward)
        self.links.append(link)
        self.state.append(feature)

    def clear_memory(self):
        self.rewards = []
        self.links = []
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
    
    def calc_loss(self): #cal loss function  #要改
        print("執行過loss")

        temp = []
        for i in range(len(self.state[0][0])):
            temp.append(0)
        temp = torch.tensor(temp)
        temp = torch.unsqueeze(temp,0)
        for i in range(len(self.state)-1):
            for j in range(len(self.state[i]),len(self.state[-1])):
                self.state[0] = torch.cat((self.state[i],temp),0)
        
        state = torch.tensor([item.detach().numpy() for item in self.state])

        reward = self.calc_R(state)

        _ , critic = self.forward(state)
        acc = critic.squeeze()
        critic_loss = (reward - acc) 
        """ 用不到原因：我們的action並不固定，並不是像遊戲裡有固定的上下左右可以選擇，每次選取完link之後，被選取的link都不會再被重複選取，因此不能使用下面的code
        probs = torch.sigmod(actor, dim = 1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(reward - values)
        """
        total_loss = critic_loss
        return total_loss

if __name__=="__main__":
    lr = 1e-4
    #example
    state = [[[0.25,0.67],[0.44,0.11]],[[0.43,0.69],[0.25,0.67],[0.44,0.11]]]
    input_dims = 2
    n_links = 1
    global_actor_critic = ActorCritic(input_dims ,n_links)
