# -*- coding:utf-8 -*-
"""
贪吃蛇
"""
import pygame
import sys

import math
from collections import namedtuple
import time
from pygame.locals import *

from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


import os 
os.environ['SDL_VIDEODRIVER'] = 'dummy'  #如要显示游戏界面则注释这一行
class Tanchishe:
# 定义颜色变量
    def __init__(self):
        self.black_colour = pygame.Color(100, 100, 100)
        self.white_colour = pygame.Color(255, 20, 30)
        self.red_colour = pygame.Color(1,1 , 1)
        self.grey_colour = pygame.Color(255, 255, 255)
        
        
        self.count = 0
        self.ftpsClock = pygame.time.Clock()
        # print(ftpsClock)
        # 创建一个窗口
        self.gamesurface = pygame.display.set_mode((260, 260))
        # print(gamesurface)
       # 设置窗口的标题
        
        # 初始化变量
        # 初始化贪吃蛇的起始位置
        self.snakeposition = [100, 100]
        # 初始化贪吃蛇的长度
        self.snakelength = [[100, 100], [80, 100], [60, 100]]
        # 初始化目标方块的位置
        self.square_purpose = [100, 200]
        # 初始化一个数来判断目标方块是否存在
        self.square_position = 1
        # 初始化方向，用来使贪吃蛇移动
        self.derection = "down"
        self.change_derection = self.derection
        self.endgame = 0
        pygame.display.flip()
        self.screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        # return screen_image
    # 定义游戏结束函数
    def gameover(self):
        # 设置提示字体的格式
        self.__init__()
        # self.main()
        return self.screen_image

    # 定义主函数
    def main(self,action):
        count= 0
        done = 0
        # [u=0,d=1,l=2,r=3]
        
        # 初始化pygame，为使用硬件做准备
        # reward = 0
        reward = 0
        pygame.init()
        pygame.time.Clock()
        self.pygame = pygame
        pygame.display.set_caption('贪吃蛇')
        
        # 进行游戏主循环
        while True:
            # 检测按键等pygame事件
            
                
                
                    # 判断键盘事件,用w,s,a,d来表示上下左右
            if action == 3 :
                self.change_derection = "right"
            if action == 2:
                self.change_derection = "left"
            if action == 0:
                self.change_derection = "up"
            if action == 1:
                self.change_derection = "down"
            
            # 判断移动的方向是否相反
            if self.change_derection == 'left' and not self.derection == 'right':
                self.derection = self.change_derection
            if self.change_derection == 'right' and not self.derection == 'left':
                self.derection = self.change_derection
            if self.change_derection == 'up' and not self.derection == 'down':
                self.derection = self.change_derection
            if self.change_derection == 'down' and not self.derection == 'up':
                self.derection = self.change_derection
            else:
                reward = -1
                self.endgame = 1
            # 根据方向，改变坐标
            if self.derection == 'left':
                self.snakeposition[0] -= 20
            if self.derection == 'right':
                self.snakeposition[0] += 20
            if self.derection == 'up':
                self.snakeposition[1] -= 20
            if self.derection == 'down':
                self.snakeposition[1] += 20
            # 增加蛇的长度
            self.snakelength.insert(0, list(self.snakeposition))
            # 判断是否吃掉目标方块
            if self.snakeposition[0] == self.square_purpose[0] and self.snakeposition[1] == self.square_purpose[1]:
                self.square_position = 0
                reward = 1
                self.count+=1
            else:
                reward=-0.5
                self.snakelength.pop()
            # 重新生成目标方块
            if self.square_position == 0:
                # 随机生成x,y,扩大二十倍，在窗口范围内
                x = random.randrange(1, 13)
                y = random.randrange(1, 13)
                self.square_purpose = [int(x * 20), int(y * 20)]
                self.square_position = 1
            # 绘制pygame显示层
            self.gamesurface.fill(self.grey_colour)
            for position in self.snakelength:
                pygame.draw.rect(self.gamesurface, self.white_colour, Rect(position[0], position[1], 20, 20))
                pygame.draw.rect(self.gamesurface, self.red_colour, Rect(self.square_purpose[0], self.square_purpose[1], 20, 20))
            # 刷新pygame显示层
            # pygame.display.flip()
            # 判断是否死亡
            
            if self.snakeposition[0] < 0 or self.snakeposition[0] > 260:
                
                reward = -1
                self.endgame = 1
            if self.snakeposition[1] < 0 or self.snakeposition[1] > 260:
                
                reward = -1
                self.endgame =1
            
            for snakebody in self.snakelength[1:]:
                if self.snakeposition[0] == snakebody[0] and self.snakeposition[1] == snakebody[1]:
                    
                    reward = -1
                    self.endgame =1 
            SIZE = 20
            for x in range(SIZE, 260, SIZE):
                pygame.draw.line(self.gamesurface, self.white_colour, (x, 0), (x, 260), 1)
        # 画网格线 横线
            for y in range(SIZE, 260, SIZE):
                pygame.draw.line(self.gamesurface, self.white_colour, (0, y), (260, y), 1)   
            # 控制游戏速度
            screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
            # print(self.snakeposition)
            self.ftpsClock.tick(100000)
            pygame.display.update()
            return torch.from_numpy(np.array(reward)).reshape(1,1),screen_image,self.endgame,self.count

        
resize = T.Compose([T.ToPILImage(),
                    
                    T.Resize(30, interpolation=Image.CUBIC),
                    
                    T.ToTensor()])

def get_screen(screen):
    
    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen)
    
    screen = resize(screen)
    
    return screen.unsqueeze(0).to(device)

LR = 0.001                   # 学习率
EPSILON = 0.8              # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.9                 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
N_ACTIONS = 4  # 棋子的动作0，1，2，3

BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

steps_done = 0

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward','next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.counter = 0
        
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        self.counter+=1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(128, 4)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 一层卷积
        x = F.relu(self.bn2(self.conv2(x)))  # 两层卷积
#         print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))  # 三层卷积
#         print(x.view(x.size(0), -1).shape)
        return self.head(x.view(x.size(0), -1))  # 全连接层
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device) #DQN需要使用两个神经网络  #eval为Q估计神经网络 target
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=LR) # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式
        self.count = 0
    def select_action(self,x):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        print(eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.eval_net.forward(x).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
    def choose_action(self, x):
        
        x = torch.FloatTensor(x).to(device)
        # 这里只输入一个 sample,x为场景
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x) #将场景输入Q估计神经网络
            #torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
            action = torch.max(actions_value, 1)[1].data # 返回动作最大值
            # print(torch.max(actions_value.data, 1)[1])
        else:   # 选随机动作
            action = torch.from_numpy(np.array([np.random.randint(0, N_ACTIONS)])) # 比如np.random.randint(0,2)是选择1或0
        
        # print(action)
        return action.to(device)
    
    def learn(self):
        
        
        
        # print(1)
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        b_s = torch.cat(batch.state).to(device)
        # print(b_s.shape)
        b_a = torch.cat(batch.action).reshape([128,1]).to(device)
        
        b_r = torch.cat(batch.reward).to(device)
        b_s_ = torch.cat(batch.next_state).to(device)
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        # start_time = time.time()
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach Q现实
        q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1) DQL核心公式
        loss = self.loss_func(q_eval, q_target.float()) #计算误差
        # 计算, 更新 eval net
        self.optimizer.zero_grad() #
        loss.backward() #反向传递
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.count == 1000:
            print("保存模型")
            torch.save(self.eval_net.state_dict(),"tanchishe_DQN1.pkl")
            torch.save(self.target_net.state_dict(),"tanchishe_DQN2.pkl")
            self.count =0
        self.count+=1
        return loss
        # end_time = time.time()
        # print("耗时",end_time-start_time)
 # 定义 DQN 系统
if __name__ == "__main__":
    device = torch.device("cuda:0")
    false=0
    succes = 0
    env = Tanchishe()
    study=1
    num=0
    dqn = DQN()
   
    memory = ReplayMemory(20000)
    for i in range(1000000):
       s = env.gameover().transpose(1,0,2)

       s = get_screen(s.transpose(2,0,1))
       
       while True:
        
            a = dqn.choose_action(s.cpu())
#            print(a)
            reward,screen_image,done,count = env.main(a)

            screen_image = screen_image.transpose(1,0,2)
            screen_image = get_screen(screen_image.transpose(2,0,1))
#             print(screen_image.shape)
#             plt.imshow(screen_image.cpu().numpy()[0,:,:,:].transpose(1,2,0))
#             plt.show()
            memory.push(s, a, reward, screen_image)
            
            if memory.counter > memory.capacity:
                # print("需要学习")
                if study==1:
                    
                    study=0
                # start_time = time.time()
                loss = dqn.learn()
                print(loss)
            if done==1 or done==2:    # 如果回合结束, 进入下回合
                #print(loss1)
                if done==1:
                    
                    
                    false+=1
                if done==2:
                    
                    succes+=1
                                 
                break
            s = screen_image
       if i%100==0 and i !=0:
            num = num/100
            print("现在是第 {} 代，蛇蛇吃下了 {} 个方块".format(i,num))
            num = 0
       else:
            num+=count
        # end_time = time.time()
        # print("耗时",end_time-start_time)
 # 定义 DQN 系统

            
            
 