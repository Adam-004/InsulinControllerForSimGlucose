from pathlib import Path
from PIL import ImageGrab
import imageio
import time
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simglucose.envs import T1DSimGymnaisumEnv
from datetime import datetime, timedelta

class CustomT1DSimGymnaisumEnv(T1DSimGymnaisumEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_time = datetime(2025, 1, 1, 0, 0, 0)#Szimuláció kezdő ideje éjfél

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]  
        target_bg = 120  
        bg_tolerance = 20         
        self.current_time += timedelta(minutes=3)
        deviation = abs(blood_glucose - target_bg)

        #00:00 és 06:00 közötti szigorúbb bünti       
        if self.current_time.hour < 6:
            if blood_glucose > 160:
                reward -=15
            elif 100 <= blood_glucose <=150:
                reward += 7
        
        if blood_glucose > target_bg + bg_tolerance:
            reward -= 5 * (deviation / 10) 
        elif blood_glucose < target_bg - bg_tolerance:
            reward -= 5 * (deviation / 10) 
        else:
            reward += 5 

            if hasattr(self, 'last_blood_glucose'):
                fluctuation = abs(blood_glucose - self.last_blood_glucose)
                if fluctuation < 10: 
                    reward += 2  
                elif fluctuation >= 10 and fluctuation < 20:  
                    reward -= 1  
            
            self.last_blood_glucose = blood_glucose

        
        print(f"[{self.current_time.strftime('%H:%M')}] Blood Glucose: {blood_glucose}, Reward: {reward}")

        return observation, reward, terminated, truncated, info
 
class LowGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        reward = -0.1  # Small penalty for every step to encourage action

        # Goal: Avoid hypoglycemia (< 70 mg/dL) at all costs.
        if blood_glucose < 70:
            deviation = 70 - blood_glucose
            reward -= 10 + (deviation ** 1.5) # Severe, escalating penalty for being low
        elif blood_glucose < 80:
            # Warning zone penalty
            reward -= 5
        elif blood_glucose > 180:
            # Mild penalty for being high, as this is not the primary focus
            reward -= (blood_glucose - 180) / 20
        else: # 80 <= BG <= 180
            # Reward for being in a safe, non-low range
            reward += 1.0

        # Penalize any insulin action when BG is low or dropping towards low
        if action[0] > 0 and blood_glucose < 100:
            reward -= 20 * action[0] # Strong penalty for incorrect action

        return observation, reward, terminated, truncated, info

class HighGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        reward = -0.1 # Small penalty for every step to encourage action

        # Goal: Bring high glucose down, avoid going > 180 mg/dL
        if blood_glucose > 180:
            deviation = blood_glucose - 180
            reward -= 5 + (deviation / 10) # Strong penalty for being high
        elif blood_glucose < 70:
            # Also penalize hypoglycemia, though it's not the main focus
            reward -= 15
        else: # 70 <= BG <= 180
            # Reward for being in the target range
            reward += 1.0

        # Penalize inaction when BG is high
        if action[0] == 0 and blood_glucose > 160:
            reward -= 10

        return observation, reward, terminated, truncated, info

class InnerGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        reward = -0.1 # Small penalty for every step to encourage action

        # Goal: Maintain tight control within 70-130 mg/dL
        target_bg = 100
        if 70 <= blood_glucose <= 130:
            # Quadratic reward, highest at the target_bg
            reward += 2.0 - 0.005 * ((blood_glucose - target_bg) ** 2)
        elif blood_glucose < 70:
            reward -= 15 + (70 - blood_glucose) # Strong penalty for hypo
        else: # blood_glucose > 130
            reward -= 5 + (blood_glucose - 130) / 10 # Strong penalty for hyper

        # Penalize large insulin doses to encourage fine-tuning
        if action[0] > 0.5:
            reward -= 5 * action[0]

        return observation, reward, terminated, truncated, info
    