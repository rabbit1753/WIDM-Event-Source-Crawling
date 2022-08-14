import numpy as np
import random
import torch as t
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


criterion = nn.CrossEntropyLoss()
gamma = 0.9
actor_lr = 0.001
critic_lr = 0.01
epochs = 500

class share_layer(nn.Module):
    def __init__(self):
        
        super(share_layer, self).__init__()
        self.linear1 = nn.Linear(4, 20)
        nn.init.normal_(self.linear1.weight, 0, 0.1)
        nn.init.constant_(self.linear1.bias, 0.1)

    def forward(self, out):
        
        out = self.linear1(out)
        out = F.relu(out)

        return out
    
class Actor(nn.Module):
    def __init__(self, sl):
        
        super(Actor, self).__init__()

        self.share_layer = sl

        self.linear2 = nn.Linear(20, 2)
        nn.init.normal_(self.linear2.weight, 0 ,0.1)
        nn.init.constant_(self.linear2.bias, 0.1)

    def forward(self, x):
        
        out = t.from_numpy(x).float()

        out = self.share_layer(out)
        out = self.linear2(out)
        prob = F.softmax(out, dim = 1)
        return prob, out

class Critic(nn.Module):

    def __init__(self, sl):

        super(Critic, self).__init__()

        self.share_layer = sl

        self.linear2 = nn.Linear(20, 1)
        nn.init.normal_(self.linear2.weight, 0, 0.1)
        nn.init.constant_(self.linear2.bias, 0.1)
    
    def forward(self, x):

        out = t.from_numpy(x).float()

        out = self.share_layer(out)

        out = self.linear2(out)

        return out

def choose_action(prob):
    
    action = np.random.choice(a = 2, p = prob[0].detach().numpy())
    return action

def Actor_learn(optim, critic, s, s_, a, r, logits):

    # s : 當前狀態
    # s_ : 下一個狀態
    # a : 當前執行的動作
    # r : reward

    V_s = critic(s)
    V_s_ = critic(s_)

    a = t.tensor([a]).long()
    logp_a = criterion(logits, a)

    l = r + V_s - V_s_
    l = l.item()
    loss = l * logp_a

    optim.zero_grad()

    loss.backward()
    optim.step()

def Critic_learn(optim, critic, s, r, s_):

    V_s = critic(s)
    V_s = critic(s_)

    loss = (r + gamma * V_s_.item() - V_s)**2
    optim.zero_grad() # 清空上一輪的gradient，
    loss.backward() # 

    optim.step()

    return r + gamma * V_s_.item() - V_s

def learn():

    sl = share_layer()
    actor = Actor(sl)
    critic = Critic(sl)

    actor.train()
    critic.train()

    actor_optim = t.optim.Adam(actor.parameters(), lr = actor_lr)
    critic_optim = t.optim.Adam(critic.parameters(), lr = critic_lr)
    train_reward = []

    for i in range(epochs):

        # state = env.reset()
        done = False
        sum_rewards_i = 0
        step = 0

        while not done:

            step += 1

            # env.render()
            state = np.expand_dims(state, axis = 0)

            prob, logits = actor(state)

            action = choose_action(prob)

            # state_, reward, done, info = env.step(action)
            # if done:
            #     reward = -20.0
            sum_rewards_i += reward

            l = Critic_learn(critic_optim, critic, state, reward, state_)
            Actor_learn(actor_optim, criticm, state, state_, action, reward, logits)
            state = state_

        train_reward.append(sum_rewards_i)
        print("epoch is :", i)
        print("step nums is :", step)

if __name__=="__main__":

    learn()
    # env.close()


