from Worker import Agent
from Global_Network import ActorCritic,ShareAdam
import torch.multiprocessing as mp

if __name__ == '__main__':

    lr = 1e-4
    input_dims = 768
    n_links = 20
    gamma = 0.8
    global_actor_critic = ActorCritic(input_dims, n_links, gamma)
    global_actor_critic.share_memory()
    optim = ShareAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))

    # workers = [Agent(input_dims,global_actor_critic,
    #                 optim,
    #                 n_links,
    #                 gamma,
    #                 ) for i in range(mp.cpu_count())]
    workers = [Agent(input_dims,global_actor_critic,
                    optim,
                    n_links,
                    gamma,)]
    [w.start() for w in workers]
    [w.join() for w in workers]