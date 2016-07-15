#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Example for using hyperparameter optimization (hpo) package.

In this example, we will try to optimize a function of
2 variables (branin) using both hpo and grid search. 

"""

import brainiak.hyperparamopt.hpo as hpo
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Branin is the function we want to minimize.
# It is a function of 2 variables.
# In the range x1 in [-5, 10] and x2 in [0, 15],
# this function has 2 local minima and 1 global minima.
# Global minima of -16.6 at (-3.7, 13.7). 
# This is the modified version (Branin-Hoo) of the standard branin function.
# If you want the standard version (which has 3 global minima),
# you can omit the "+5*x1" term at the end
# For more details, see http://www.sfu.ca/~ssurjano/branin.html
def branin(x1, x2):
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8*np.pi)
    return a*((x2 - b*x1*x1 + c*x1 - r)**2) + s*(1-t)*np.cos(x1) + s + 5*x1

# This is a wrapper around branin that takes in a dictionary
def branin_wrapper(args):
    x1 = args['x1']
    x2 = args['x2']
    return branin(x1,x2)

# Define ranges for the two variables
x1lo = -5
x1hi = 10
x2lo = 0
x2hi = 15

##############################
# Optimization through hpo
##############################

# Define a space for hpo to use
# The space needs to define
# 1. Name of the variables
# 2. Default samplers for the variables (use scipy.stats objects)
# 3. lo and hi ranges for the variables (will use -inf, inf if not specified)
space = {'x1':{'dist': st.uniform(x1lo, x1hi-x1lo), 'lo':x1lo, 'hi':x1hi},
         'x2':{'dist': st.uniform(x2lo, x2hi-x2lo), 'lo':x2lo, 'hi':x2hi}}

# The trials object is just a list that stores the samples generated and the
# corresponding function values at those sample points.
trials = []

# Maximum number of samples that will be generated.
# This is the maximum number of function evaluations that will be performed.
n_hpo_samples = 100

# Call the fmin function that does the optimization.
# The function to be optimized should take in a dictionary. You will probably
# need to wrap your function to do this (see branin() and branin_wrapper()).
# You can pass in a non-empty trials object as well e.g. from a previous
# fmin run. We just append to the trials object and will use existing data
# in our optimization.
print("Starting optimization through hpo")
best = hpo.fmin(loss_fn=branin_wrapper, space=space,
                max_evals=n_hpo_samples, trials=trials)

# Print out the best value obtained through HPO
print("Best obtained through HPO (", n_hpo_samples, " samples) = ",
       best['x1'], best['x2'], "; min value = ", best['loss'])

#####################################
# Optimization through grid search
#####################################

# Divide the space into a uniform grid (meshgrid)
n = 200
x1 = np.linspace(x1lo, x1hi, n)
x2 = np.linspace(x2lo, x2hi, n)
x1_grid, x2_grid = np.meshgrid(x1, x2)

# Calculate the function values along the grid
print("Starting optimization through grid search")
z = branin(x1_grid, x2_grid)

# Print out the best value obtained through grid search
print("Best obtained through grid search (", n*n, " samples) = ",
       x1_grid.flatten()[z.argmin()], x2_grid.flatten()[z.argmin()],
       "; min value = ", z.min())

########
# Plots
########

# Convert trials object data into numpy arrays
x1 = np.array([tr['x1'] for tr in trials])
x2 = np.array([tr['x2'] for tr in trials])
y = np.array([tr['loss'] for tr in trials])

# Plot the function contour using the grid search data
h = (z.max()-z.min())/25
plt.contour(x1_grid, x2_grid, z, levels=np.linspace(z.min()-h, z.max(), 26))

# Mark the points that were sampled through HPO
plt.scatter(x1, x2, s=10, color='r', label='HPO Samples')

# Mark the best points obtained through both methods
plt.scatter(best['x1'], best['x2'], s=30, color='b', label='Best HPO')
plt.scatter(x1_grid.flatten()[z.argmin()], x2_grid.flatten()[z.argmin()],
            s=30, color='g', label='Best grid search')

# Labels
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Hyperparameter optimization using HPO (Branin function)')
plt.legend()
plt.show()

