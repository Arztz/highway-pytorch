import argparse
import datetime
from graph_utils import ShowGraph
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from ddqn import DDQN

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch.nn as nn
import os
import matplotlib
import psutil
import highway_env
from gymnasium.vector import AsyncVectorEnv
import time
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
g = ShowGraph(log_dir=f'runs/highway-0')
torch.set_num_threads(22)
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ENVS = 14
start = time.time()

def make_env():
    def thunk():
        return gymnasium.make("highway-v0", render_mode=None)
    return thunk
class Agent:
    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_set]

        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.env_make_params = hyperparameters.get('env_make_params',{})
        self.pretrained_model = hyperparameters.get('pretrained_model', None)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None


        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')



    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states,actions,new_states,rewards,terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        states = states.view(states.size(0), -1)
        actions = torch.stack(actions).to(device)   
        new_states = torch.stack(new_states).to(device)
        new_states = new_states.view(new_states.size(0), -1)
        rewards = torch.stack(rewards).to(device)   
        terminations = torch.as_tensor(terminations).float().to(device)
        # print("states.shape:", states.shape)
        # print("actions.shape:", actions.shape)
        # print("new_states.shape:", new_states.shape)
        # print("rewards.shape:", rewards.shape)
        # print("terminations.shape:", terminations.shape)
        # print("target_dqn(new_states).shape:", target_dqn(new_states).shape)
        # print("best_actions_from_policy.shape:", best_actions_from_policy.shape)
        # print("gather result:", target_q.shape)
        with torch.no_grad():
            best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
            target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).gather(dim=1,index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()


        current_q = policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def log_message(self,str):
        print(str)
        with open(self.LOG_FILE, 'a') as log_file:
            log_file.write(str+ '\n') 

    def run(self,is_training=True,render=False):

        if is_training:
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time


            self.log_message(f"Double DQN Enable")
            self.log_message(f"Start time: {start_time}")

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        # env = gymnasium.make("highway-v0", render_mode='human' if render else None, **self.env_make_params)
        
        env = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

        obs_space = env.single_observation_space
        num_states = obs_space.shape[0] if len(obs_space.shape) == 1 else np.prod(obs_space.shape)
        num_actions = env.single_action_space.n
        reward_per_episode = []
        epsilon_history = []
        optimize_every_n_steps = 10
        print(f'state: {num_states}  action: {num_actions}')
        policy_dqn = DDQN(num_states,num_actions).to(device)

        if is_training:
            if self.pretrained_model and os.path.exists(self.pretrained_model):

                print(f"Loading pretrained model from {self.pretrained_model}")
                policy_dqn.load_state_dict(torch.load(self.pretrained_model))
                print(f"Loaded pretrained model from {self.pretrained_model}")
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DDQN(num_states,num_actions  ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            best_reward = -9999999
        else:
            # Load the model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        #  Training loop
        for episode in itertools.count():
            
            state, _ = env.reset()
            state = torch.as_tensor(state,dtype=torch.float,device=device).view(NUM_ENVS, -1)

            # terminated = False
            # episode_reward = 0.0
            episode_reward = np.zeros(NUM_ENVS)
            terminated = np.zeros(NUM_ENVS, dtype=bool)
            max_timesteps = 1000
            steps = 0
            while (not  np.all(terminated) and np.max(episode_reward) < self.stop_on_reward):
                steps += 1
                action=[]
                # Next action:
                # (feed the observation to your agent here)
                for i in range(NUM_ENVS):
                    if is_training and random.random() < epsilon:
                        act = env.single_action_space.sample()
                        # act = torch.as_tensor(act,dtype=torch.long,device=device)
                    else:
                        with torch.no_grad():
                            q=policy_dqn(state[i].unsqueeze(0))
                            act = q.argmax(dim=1).item()
                    action.append(act)
                # Processing:
                action = np.array(action, dtype=np.int32)
                new_state, reward, terminated, _, info = env.step(action)
                print("Step time:", time.time() - start)
                # print(terminated)
                # shaped_reward = reward
                # if max_timesteps < steps:
                #     shaped_reward -= 5
                #     terminated = True
                # else:
                #     shaped_reward -= 0.01
                # if new_state[3] < 0:  # vel_y < 0 (ลง)
                #     shaped_reward += 0.02

                # # ถ้า lander ยกตัวขึ้น = ลงโทษเล็กน้อย
                # if new_state[3] > 0:  # vel_y > 0 (ขึ้น)
                #     shaped_reward -= 0.2

                # # ถ้าแตะพื้น = ให้รางวัลเยอะๆ
                # if not landed_bonus_given and (new_state[6] > 0.5 or new_state[7] > 0.5):
                #     shaped_reward += 10.0
                #     landed_bonus_given = True
                # episode_reward += shaped_reward

                new_state = torch.as_tensor(new_state,dtype=torch.float,device=device).view(NUM_ENVS, -1)
                reward = torch.as_tensor(reward,dtype=torch.float,device=device)
                
                for i in range(NUM_ENVS):
                    episode_reward[i] += reward[i].item()
                    if is_training:
                        memory.append((state[i].detach(),torch.tensor(action[i], dtype=torch.long, device=device),new_state[i].detach(),reward[i].detach(),terminated[i]))
                
                # step_count += 1
                state = new_state
                for i in range(NUM_ENVS):
                    if terminated[i]:
                        episode += 1
                        # if max_timesteps>steps[i]:
                        #     steps = 0
                        max_episode_reward = episode_reward[i]
                        reward_per_episode.append(max_episode_reward)
                        if episode % 50 == 0:
                            print(f'{datetime.datetime.now().strftime(DATE_FORMAT)} Episode: {episode} reward {max_episode_reward}')
                        if is_training:
                            if max_episode_reward > best_reward:
                                self.log_message(f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {max_episode_reward} at episode {episode}")
                                self.log_message(f"Memory used: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
                                
                                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                                best_reward = max_episode_reward

                            current_time = datetime.datetime.now()
                            if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                                g.log(reward_per_episode, epsilon)
                                last_graph_update_time = current_time
                            if len(memory) > self.mini_batch_size:
                                                        #sample from memory
                                mini_batch = memory.sample(self.mini_batch_size)
                                self.optimize(mini_batch,policy_dqn,target_dqn)
                                epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)
                                epsilon_history.append(epsilon)
                                
                                if step_count > self.network_sync_rate:
                                    target_dqn.load_state_dict(policy_dqn.state_dict())
                                    step_count = 0
                    
        env.close() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help='Training mode',action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameters_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True,render=False)
    else:
        dql.run(is_training=False,render=True)