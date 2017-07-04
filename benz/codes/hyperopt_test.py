from hyperopt import hp
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
    
    
space =  hp.choice('c2', list(frange(1,3,0.1)))

import hyperopt.pyll.stochastic
print (hyperopt.pyll.stochastic.sample(space))