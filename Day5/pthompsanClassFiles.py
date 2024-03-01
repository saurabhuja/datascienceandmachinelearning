# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:16:49 2024
Thompsan Sampling
@author: tsecl
"""

# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# =============================================================================
# Variable Declarations
# =============================================================================
simulation_dataset = pd.read_csv("Ads ClickThroughRate.csv")
d = 10      # number of Ads
N = 10000   # number of rounds

number_of_times_reward_0 = [0]*d
number_of_times_reward_1 = [0]*d

ad_displayed = []

# =============================================================================
# Ad Choosing Logic
# =============================================================================

for n in range(N):
    max_dist=0
    for i in range(d):
        
        current_dist = random.betavariate(number_of_times_reward_1[i]+1, 
                                          number_of_times_reward_0[i]+1)
        
        if current_dist>max_dist:
            max_dist = current_dist
            ad = i
            
    ad_displayed.append(ad)
    
    clicked_reward = simulation_dataset.values[n,ad]
    
    if clicked_reward==1:
        number_of_times_reward_1[ad] += 1
    else:
        number_of_times_reward_0[ad] += 1
        
plt.hist(ad_displayed)  
plt.show()      
    

































