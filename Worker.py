import numpy as np
import json
import random
import torch as t
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn import functional as F
import matplotlib.pyplot as plt
from Global_Network import ActorCritic
import Crawler.FeaEx as FeaEx
import Crawler.Crawler as crawler
import job_assignment as job
import URL_Frontier_code
import requests 
import dis_URL_code

event_source_url = []
frontier = URL_Frontier_code.URL_Frontier()

class Agent(mp.Process):
    def __init__(self, input_dims, global_actor_critic, optimizer, n_links, gamma):
        super(Agent, self).__init__()

        self.local_actor_critic = ActorCritic(input_dims, n_links, gamma)
        self.global_actor_critic = global_actor_critic
        self.episode_idx = 1
        self.optimizer = optimizer

    def run(self):
        stop = 4
        while self.episode_idx < stop:
            round_idx = 1
            seed = job.job_assignment(round_idx)
            first_round = True
            while True: # score x probability < 一個值
                if not first_round:              
                    action = frontier.return_link()
                    a_index = round_idx+2
                else:   
                    action = seed
                
                print(action)
                page = crawler.web_contain(action)
                page_reward = dis_URL_code.dis_URL(seed, action)

                if page_reward > 0.7:   # 隨便設的門檻，若大於門檻值就加入到event_source_url
                    event_source_url.append(action)

                feature_vector, links = FeaEx.conclu(page)
                state_ = feature_vector
                state_t = t.from_numpy(state_) 
                probability, score = self.local_actor_critic.forward(state_t)
                
                link_list = []
                for l, f, p, s in zip(links,feature_vector,probability,score):
                    tmp = []
                    tmp.append(l)
                    tmp.append(p)
                    tmp.append(s)
                    tmp.append(f)
                    link_list.append(tmp)
                print(len(link_list))
                frontier.process_list(link_list) # 把這一輪 page 的 link 的各項資訊加入 frontier
                if not first_round: # page_reward 這一輪點選的 page 是1分還是0分、a_index 是點選的 page 的 url_index
                    self.local_actor_critic.record_episode(page_reward, a_index, state_t) # state_t 是這一個 page 所有可點 link 的 feature 態
                    print('Round:',round_idx,'reward %.1f' % page_reward)
            
                first_round = False
                round_idx += 1
                if frontier.discriminate() == False or round_idx == 10:
                    break
                
            loss = self.local_actor_critic.calc_loss()
            self.optimizer.zero_grad()  # 先清空上一輪的梯度為0
            loss.backward() # 反向回饋

            for local_param, global_param in zip(   # 每個A3C版本都有出現，阿但是在幹麻還沒懂
                self.local_actor_critic.parameters(), 
                self.global_actor_critic.parameters()):
                global_param = local_param
            self.optimizer.step()   # 更新梯度
            self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict()) # 複製 global 參數 到 local 去
            self.local_actor_critic.clear_memory()
            

