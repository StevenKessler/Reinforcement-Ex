#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
from gym.spaces import Tuple, Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym_minigrid.minigrid import *
import numpy as np


class SimpleMazeEnv(MiniGridEnv):
    
    def __init__(self):
        super().__init__(width=11, height=8, max_steps=1000)
        
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        self.grid.horz_wall(0,0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        
        self.grid.vert_wall(3, 2, 3)
        self.grid.vert_wall(8, 1, 3)
        self.put_obj(Wall(), 6, 5)
        
        self.agent_pos = (1, 4)
        self.grid.set(*self.agent_pos, None)
        self.agent_dir = 0
        
        self.put_obj(Goal(), 9, 1)
        
        self.mission = 'Reach the goal'
        
        
class PosObsWrapper(gym.ObservationWrapper):
    
    def __init__(self, env):
        super(PosObsWrapper, self).__init__(env)
        self.observation_space = Tuple((
                MultiDiscrete([env.width, env.height]),
                Discrete(4)))
        
    def observation(self, observation):
        return (tuple(self.agent_pos), self.agent_dir)
    
        
class LimitActionWrapper(gym.ActionWrapper):
    
    def __init__(self, env, n):
        assert type(env.action_space) == gym.spaces.Discrete
        assert env.action_space.n >= n
        super(LimitActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(n)
        
    def action(self, action):
        if action not in self.action_space:
            raise ValueError('Invalid action')
        return action
    
    
def simple_maze_env():
    env = SimpleMazeEnv()
    env = PosObsWrapper(env)
    env = LimitActionWrapper(env, 3)
    return env