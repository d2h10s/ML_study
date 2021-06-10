#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:06:55 2021

@author: you know who.
"""


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class gridworld:
    
    def __init__(self, size):
        
        self.size        = size
        self.state       = [(i,j) for i in range(self.size) for j in range(self.size)]
        self.goal        = (self.size -1, self.size -1)
        self.action_list = [(1,0),(-1,0),(0,-1),(0,1)] #동서남북 
        self.reward    = {}
        self.avail_act = {}
        self.current_state = (0,0)
        
        for st in self.state:
            self.reward[st]      = -1  # 모든 이동에 -1 리워드 
            self.avail_act[st]   = self.action_list[:] #동서남북 네 방향 액션 가능
            if st[0]==0: #left move impossible
                self.avail_act[st].remove((-1,0))
            if st[0]==self.size-1: #right move impossible
                self.avail_act[st].remove((1,0))
            if st[1]==0: #down move impossible
                self.avail_act[st].remove((0,-1))
            if st[1]==self.size -1: #up move impossible
                self.avail_act[st].remove((0,1))
            if st==self.goal : #goal_state
                self.avail_act[st] = []
            
        # 그리드 월드 렌더링을 위한 파라메터 
        self.xs=[st[0] for st in self.state]
        self.ys=[st[1] for st in self.state]

        self.color_data =[0 for i in range(self.size**2)]
        self.color_data[self.size**2-1] = 1 #목적지는 다른 색
    
    
    def set_huddle(self, num_ob):
        holes=np.random.choice(range(1,self.size**2-1) ,size=num_ob)

        for jj in holes:
            self.reward[self.state[jj]]=-10
            self.color_data[jj]=2
    
    def render(self, AA=None): 
        
        clear_output(True)
        
        self.fig = plt.figure(figsize=(self.size-2, self.size-2))
        self.fig.suptitle('Grid World', fontsize=14, fontweight='bold')

        self.gw = self.fig.add_subplot(111)
        self.gw.grid(color = 'k', linestyle = '--', linewidth = 0.5)
        self.gw.scatter(self.xs,self.ys, c=self.color_data, s=60)
        self.gw.axis([-0.2,self.size-0.6,-0.2,self.size-0.6])
        self.gw.plot([self.current_state[0]],[self.current_state[1]], 'r*', markersize =20)
        if AA!=None:
            for j in self.state:
                self.gw.text(j[0]+0.1,j[1]+0.1,np.round(AA[j],3),fontsize=10)
        
        plt.show()
    
    
    def reset(self, start = (0,0)):
        self.current_state = start
        self.donemark      = False
        return self.current_state
        
    def sample(self):
        action_list = self.avail_act[self.current_state]
        act = np.random.choice(len(action_list), 1)[0]
        return act
        
    def step(self, action):
        action_list = self.avail_act[self.current_state]
        action  = action_list[action]
        self.current_state = np.array(self.current_state) + np.array(action)
        self.current_state = tuple(self.current_state)
        if self.current_state == self.goal:
            self.donemark = True
        return self.current_state, self.reward[self.current_state], self.donemark, None
    


    
    
        
    
