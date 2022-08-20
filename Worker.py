import numpy as np
import random
import torch as t
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import event_source_estimator

event_source_url = []

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer):
        super(Agent, self).__init__()

        self.local_actor_critic = ActorCritic(input_dims, n_links, gamma)
        self.global_actor_critic = global_actor_critic
        self.threshold = False

        self.optimizer = optimizer

    def run(self):  # state = 這一輪的 frontier 的 vector， action = 下一輪預計決定點擊的 link 的 vector，
                            # reward = 這一輪 page 所得到 reward(由 event source estiamtor 來評估)
        round_idx = 1
        action = job_assignment()
        first_round = True
        while not frontier.discriminate(): # score x probability < 一個值
            if not first_round:              
                action = frontier.return_link()
            else:   
                first_round = False
            page = crawler(action)
            page_reward = event_source_estimator.XXXX(page)
            if page_reward > 0.7:   # 隨便設的門檻，若大於門檻值就加入到event_source_url
                event_source_url.append(action)

            links, feature_vector = feature_extraction(page)
            old_feature_vector = frontier.return_feature()
            state_ = feature_vector + old_feature_vector
            probability, score = self.local_actor_critic.forward(state_)
            
            link_list = []
            for l, f, p, s in zip(links,feature_vector,probability,score):
                tmp = []
                tmp.append(l,f,p,s)
                link_list.append(tmp)

            frontier.push(link_list)
            self.local_actor_critic.remember(state, action, page_reward)
            print('Round ',round_idx,'reward %.1f' % page_reward)
            
        loss = self.local_actor_critic.calc_loss()
        self.optimizer.zero_grad()  # 先清空上一輪的梯度為0
        loss.backward() # 反向回饋

        for local_param, global_param in zip(   # 每個A3C版本都有出現，阿但是在幹麻還沒懂
            self.local_actor_critic.parameters(), 
            self.global_actor_critic.parameters()):
            global_param = local_param
        self.optimizer.step()   # 更新梯度
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict) # 複製 global 參數 到 local 去
        self.local_actor_critic.clear_memory()
        

