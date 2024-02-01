# -*- coding: utf-8 -*-
"""
@author: Oscar Flores
@portfolio: github.com/OscarFlores-IFi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

orig_map=plt.cm.get_cmap('inferno') 
reversed_map = orig_map.reversed() 

#%%
transition_probabilities = np.array([
       [0.050, 0.000, 0.000, 0.000, 0.050, 0.050, 0.050, 0.150, 0.200, 0.200, 0.250], 
       [0.050, 0.000, 0.000, 0.000, 0.050, 0.050, 0.050, 0.150, 0.200, 0.200, 0.250], 
       [0.024, 0.024, 0.024, 0.048, 0.071, 0.071, 0.095, 0.119, 0.143, 0.167, 0.214],
       [0.031, 0.031, 0.046, 0.046, 0.046, 0.077, 0.108, 0.108, 0.154, 0.169, 0.184],
       [0.024, 0.024, 0.037, 0.037, 0.085, 0.085, 0.098, 0.098, 0.146, 0.183, 0.183],
       [0.005, 0.005, 0.020, 0.035, 0.050, 0.075, 0.106, 0.146, 0.176, 0.181, 0.201],
       [0.004, 0.008, 0.016, 0.028, 0.045, 0.069, 0.101, 0.121, 0.182, 0.202, 0.224],
       [0.007, 0.007, 0.007, 0.015, 0.029, 0.046, 0.073, 0.153, 0.221, 0.221, 0.221],
       [0.003, 0.003, 0.004, 0.006, 0.009, 0.035, 0.043, 0.104, 0.250, 0.242, 0.301],
       [0.002, 0.002, 0.002, 0.007, 0.008, 0.013, 0.023, 0.058, 0.166, 0.309, 0.410],
       [0.003, 0.003, 0.003, 0.004, 0.005, 0.012, 0.014, 0.034, 0.121, 0.214, 0.587]])

# Create heatmap
plt.imshow(transition_probabilities, cmap=reversed_map, interpolation='nearest')
plt.colorbar(label='Transition Probability')

# Add labels
plt.xticks(np.arange(11), np.arange(11))
plt.yticks(np.arange(11), np.arange(11))
plt.xlabel('To')
plt.ylabel('From')

plt.savefig('heatmap.png', dpi=300)
plt.show()

#%% 
def markov_simulation_one_individual(transition_matrix, initial_vector, T, N):
    # Validate inputs
    n = len(transition_matrix)
    if n != len(initial_vector) or not all(len(row) == n for row in transition_matrix):
        raise ValueError("Invalid input dimensions")
    # Perform simulation
    historical_states = np.zeros((T, N))
    choices = np.arange(n)
    for i in range(N):
        current_state = np.copy(initial_vector)
        for t in range(T):
            # Choose next state based on transition probabilities
            next_state = np.random.choice(choices, p=current_state)
        
            historical_states[t, i] = next_state
            # Update current state for the next iteration
            current_state = transition_matrix[next_state]
    return historical_states
initial_distribution = np.ones(11)/11 # Initial classification

np.random.seed(555)

# Set the number of iterations
T = 1000
N = 1

# Run the simulation
historical_states = markov_simulation_one_individual(transition_probabilities, initial_distribution, T, N).astype(int)
plt.plot(historical_states, 'o-')

# Add labels
plt.xticks(np.linspace(0,T,11))
plt.yticks(np.arange(11))
plt.xlabel('Time')
plt.ylabel('Category')

plt.savefig(f'path{T}.png', dpi=300)
plt.show()

#%%

def transition_matrix_calculation(input_data):
  data = input_data.copy()

  num_states = len(np.unique(input_data))
  count_matrix = np.zeros((num_states,num_states))

  for i,j in zip(data[:-1],data[1:]):
    count_matrix[i,j] += 1
  
  sum_x = pd.DataFrame(count_matrix).sum(axis=1).values

  transition_matrix = count_matrix.copy()
  for i in range(transition_matrix.shape[0]):
    transition_matrix[:,i] = transition_matrix[:,i]/sum_x

  return(pd.DataFrame(count_matrix), pd.DataFrame(transition_matrix))


plt.imshow(transition_matrix_calculation(historical_states)[1], cmap=reversed_map, interpolation='nearest')
plt.colorbar(label='Transition Probability')

# Add labels
plt.xticks(np.arange(11), np.arange(11))
plt.yticks(np.arange(11), np.arange(11))
plt.xlabel('To')
plt.ylabel('From')

plt.savefig('heatmapFrequentist.png', dpi=300)
plt.show()

#%% Estimate Markov Chain
######################
import numpy as np
from scipy.stats import dirichlet, multinomial

def simulate_multinomial_posterior(data, alpha=None, num_samples=1000):
    if alpha == None:
        alpha = np.ones(len(data)) # Dirichlet initialized in uniform distribution 
        
    # Convert data to numpy array
    observed_counts = np.array(list(data.values()))

    # Posterior distribution
    posterior = dirichlet(alpha + observed_counts)
    samples = posterior.rvs(size=num_samples)
    return samples

# Example usage

def estimate_bayesian_transition_matrix(count_matrix):
    count_dict = count_matrix.T.to_dict()
    bayesian_count_matrix = []
    
    for ii in range(len(count_dict)):
        data = count_dict[ii]
        
        samples = simulate_multinomial_posterior(data)
        M = 1000
        total_sum = np.zeros((1, len(data)))
        for i in samples:
            total_sum = total_sum + multinomial(M, i).rvs()
        bayesian_count_matrix.append(total_sum[0])

    bayesian_count_matrix = pd.DataFrame(bayesian_count_matrix)
    sum_x = pd.DataFrame(bayesian_count_matrix).sum(axis=1).values
    bayesian_transition_matrix = bayesian_count_matrix/sum_x

    return bayesian_count_matrix, bayesian_transition_matrix
    
count_matrix = transition_matrix_calculation(historical_states)[0]
bayesian_count_matrix, bayesian_transition_matrix = estimate_bayesian_transition_matrix(count_matrix)


plt.imshow(bayesian_transition_matrix, cmap=reversed_map, interpolation='nearest')
plt.colorbar(label='Transition Probability')

# Add labels
plt.xticks(np.arange(11), np.arange(11))
plt.yticks(np.arange(11), np.arange(11))
plt.xlabel('To')
plt.ylabel('From')

plt.savefig('heatmapBayesian.png', dpi=300)
plt.show()