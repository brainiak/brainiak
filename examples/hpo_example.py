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
import numpy as np
import matplotlib.pyplot as plt
import brainiak.hyperparamopt.hpo as hpo

def branin(x1, x2):
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8*np.pi)
    return a*((x2 - b*x1*x1 + c*x1 - r)**2) + s*(1-t)*np.cos(x1) + s + 5*x1

def g(args):
    x1 = args['x1']
    x2 = args['x2']
    return branin(x1,x2)

x1lo = -5
x1hi = 10
x2lo = 0
x2hi = 15

space = {'x1':{'dist':'uniform', 'lo':x1lo, 'hi':x1hi},
         'x2':{'dist':'uniform', 'lo':x2lo, 'hi':x2hi}}
trials = []
n_hpo_samples = 100

best = hpo.fmin(lossfn=g, space=space, maxevals=n_hpo_samples, trials=trials, verbose=False)
print("Best obtained through HPO (", n_hpo_samples, " samples) = ",
       best['x1'], best['x2'], "; min value = ", best['loss'])

nt = 100
x1t = np.linspace(x1lo, x1hi, nt)
x2t = np.linspace(x2lo, x2hi, nt)
x1m, x2m = np.meshgrid(x1t, x2t)
z = branin(x1m, x2m)
print("Best obtained through grid search (", nt*nt, " samples) = ",
       x1m.flatten()[z.argmin()], x2m.flatten()[z.argmin()],
       "; min value = ", z.min())

x1 = np.array([tr['x1'] for tr in trials])
x2 = np.array([tr['x2'] for tr in trials])
y = np.array([tr['loss'] for tr in trials])

h = (z.max()-z.min())/25
plt.contour(x1m, x2m, z, levels=np.linspace(z.min()-h, z.max(), 26))
plt.scatter(x1, x2, s=10, color='r', label='HPO Samples')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Hyperparamter optimization using HPO')

plt.scatter(best['x1'], best['x2'], s=30, color='b', label='Best HPO')
plt.scatter(x1m.flatten()[z.argmin()], x2m.flatten()[z.argmin()],
            s=30, color='g', label='Best grid search')
plt.legend()
plt.show()

