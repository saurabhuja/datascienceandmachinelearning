# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:41:46 2024
Upper Confidence Bound
@author: tsecl
"""

# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# =============================================================================
# Variable Declarations
# =============================================================================
simulation_dataset = pd.read_csv("Ads ClickThroughRate.csv")
d = 10      # number of Ads
N = 10000   # number of rounds
#                                Ad0  Ad1  Ad2  Ad3  Ad4
#                               [1,   0,   0,   0,   0,   0, 0, 0, 0, 0]
number_of_times_ad_displayed = [0]*d
sum_of_rewards = [0]*d
ad_displayed = []

# =============================================================================
# Ad Choosing Logic
# =============================================================================
    
for n in range(0,N):    # Loop through Rounds
    max_ucb =0
    for i in range(0,d):   # Loop through Ads
        if number_of_times_ad_displayed[i]>0:
            average_reward = sum_of_rewards[i]/number_of_times_ad_displayed[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_times_ad_displayed[i])
            current_ucb = average_reward + delta_i            
        else:
            current_ucb = 1e400   # assumed ucb to ensure Ad in initial 10 rounds gets a chance
            

        if current_ucb > max_ucb:
            max_ucb=current_ucb
            ad = i  
            
    ad_displayed.append(ad)
    number_of_times_ad_displayed[ad] = number_of_times_ad_displayed[ad]+1
    
    """ Ad ad in round n has a click or no-click """
    clicked_reward = simulation_dataset.values[n,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + clicked_reward
    
plt.hist(ad_displayed)
plt.show()    
    
            
        
               












