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

event_source_url = []
frontier = URL_Frontier_code.URL_Frontier()

class Agent(mp.Process):
    def __init__(self, input_dims, global_actor_critic, optimizer, n_links, gamma):
        super(Agent, self).__init__()

        self.local_actor_critic = ActorCritic(input_dims, n_links, gamma)
        self.global_actor_critic = global_actor_critic
        self.threshold = False

        self.optimizer = optimizer

    def run(self):  # state = 這一輪的 frontier 的 vector， action = 下一輪預計決定點擊的 link 的 vector，
                            # reward = 這一輪 page 所得到 reward(由 event source estiamtor 來評估)
        round_idx = 1
        action = job.job_assignment(round_idx)
        first_round = True
        while True: # score x probability < 一個值
            print("我跑到B拉")
            if not first_round:              
                action = frontier.return_link()
            else:   
                first_round = False
            
            print(action)
            page = crawler.web_contain(action)

            # print(action)
            # # action = "https://kktix.com/"
            # api_url = "http://140.115.54.45:8799/GetPageScore?url="
            # api_url = api_url + action
            # result = json.loads(requests.get(api_url).text)
            # page_reward = float(result["score"])
            # print("page reward：",page_reward)

            page_reward = 0.6

            if page_reward > 0.7:   # 隨便設的門檻，若大於門檻值就加入到event_source_url
                event_source_url.append(action)

            feature_vector, links = FeaEx.conclu(page)
            state_ = feature_vector
            state_t = t.from_numpy(state_) 
            probability, score = self.local_actor_critic.forward(state_t)
            print("產生狀態、連結、機率、分數")
            
            link_list = []
            for l, f, p, s in zip(links,feature_vector,probability,score):
                tmp = []
                tmp.append(l)
                tmp.append(p)
                tmp.append(s)
                tmp.append(f)
                link_list.append(tmp)
            
            frontier.process_list(link_list)
            self.local_actor_critic.record_episode(page_reward, action, state_t)
            print('Round:',round_idx,'reward %.1f' % page_reward)
            round_idx += 1
            if frontier.discriminate() == False or round_idx == 3:
                break
            
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
        

