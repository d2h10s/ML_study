#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:22:23 2021

@author: you know who
"""

import numpy as np
import matplotlib.pyplot as plt
import time



class greedypolicy:
    
    def __init__(self, env, gamma=1, V=None):
        
        self.env = env
        self.gamma = gamma
        if V ==None:
            self.V ={}
            for state in self.env.state:
                self.V[state] = 0
        else:
            self.V = V
        
           
    def policy(self, state):
        
        agent_action_list = self.env.avail_act[state]
        next_state =[tuple(np.array(state) + np.array(act)) for act in agent_action_list ]
        next_state_Q = np.zeros(len(agent_action_list))
        prob = np.ones(len(agent_action_list))
        for index, state in enumerate(next_state):
            next_state_Q[index] = self.env.reward[state] +self.gamma * self.V[state]
                 #불가능한 행동
        
        non_greedy_index = np.where( next_state_Q < np.max(next_state_Q))[0]
        for j in non_greedy_index:
            prob[j] = 0
        prob /= np.sum(prob)
        
        return prob 
        
        
        
    def action(self):
        
        prob  = self.policy(self.env.current_state)
        act  = np.random.choice(range(len(prob)), p=prob) # 확률분포에 따라 선택
        return act
    
    
    def test(self, renop=False, slow=False, start=(0,0)):
        
        self.env.reset()
        self.env.current_state = start
        d = False
        total_r = 0
        while not d:
            if renop:
                self.env.render()
                if slow:
                    time.sleep(1)
                else:
                    time.sleep(0.1)
            s, r, d, _ = self.env.step(self.action())
            total_r += r
        
        
        if renop:
            self.env.render()
        print("policy에 따라 움직였을 때 얻은 보상 {}".format(total_r))
        
        



class DP:
    def __init__(self, env, agent):
        
        self.env   = env
        self.agent = agent
        self.gamma = self.agent.gamma
        
    
    def evaluate(self, theta=1e-3):
        
        old_V = self.agent.V.copy()
        value_updated = True
        while value_updated:
            delta = 0
            new_V = {}
            
            for key in old_V:
                
                new_V[key] = 0
                
                if key != self.env.goal:
                    
                    current_state      = key[:]
                    poss_actions       = self.agent.env.avail_act[current_state]
                    prob               = self.agent.policy(current_state)
                    next_states        = [ tuple(np.array(current_state) + np.array(act)) for act in poss_actions ]
                
                    for jj in range(len(next_states)):
                        new_V[key] += prob[jj] *(self.env.reward[next_states[jj]] \
                                         + self.gamma * old_V[next_states[jj]] ) 
                delta = max(delta, abs(old_V[key]-new_V[key]))
                
            old_V = new_V.copy()
            if delta < theta:
                value_updated = False
        
        
        self.evaluatedV = new_V.copy()
        
        return self.env.render(self.evaluatedV)
    
    def improve(self):
        self.agent.V = self.evaluatedV.copy()
        print("improved")
        pass