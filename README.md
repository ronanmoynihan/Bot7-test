# Bot7 Test

Testing the [Bot7](https://github.com/j-wilson/bot7) HyperParams Bayesian Optimization module.  

### Objective Function

Currently one test which is a regression problem using the [Boston Housing toy dataset](http://lib.stat.cmu.edu/datasets/boston).  
It uses a three layer neural network.

```lua
  model:add(nn.Linear(n_inputs, numhid1)) 
  model:add(nn.Sigmoid())
  model:add(nn.Linear(numhid1, n_outputs))
```  	  

Three hyperparams are used in the objective function.  
- learningRate  
- momentum  
- learningRateDecay  

### Running the test

```
  th run_regressionTest.lua 
``` 

Once the experiment is finished it tests the hyper params on test data and prints a sample of 20 predictions.

### Results

The following are the best values I found from running manual tests which gave an MSE (Test Data) of 10,

```lua
  optimState = {
    learningRate = 1e-1,
    momentum = 0.4,
    learningRateDecay = 1e-4
  }
```  

After running a number of bot7 tests it came close to these numbers with a MSE (Test Data) ranging from 11 to 22.


