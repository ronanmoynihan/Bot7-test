# Bot7 Test

Testing the [Bot7](https://github.com/j-wilson/bot7) HyperParams Bayesian Optimization module.  

### Dependencies

- Updated version of Torch7   
- [gpTorch7](https://github.com/j-wilson/gpTorch7)  
- [bot7](https://github.com/j-wilson/bot7)

### Regression Objective Function

Currently one test which is a regression problem using the [Boston Housing toy dataset](http://lib.stat.cmu.edu/datasets/boston).  
It uses a one hidden layer neural network.

```lua
  model:add(nn.Linear(n_inputs, numhid1)) 
  model:add(nn.Sigmoid())
  model:add(nn.Linear(numhid1, n_outputs))
```  	  

#### Regression Objective test

```
  th run_regressionTest.lua 
``` 

Once the experiment is finished it tests the hyper params on test data and prints a sample of 20 predictions.

### AutoML test

Bot7 contains an automated hyperparameter example ([autoML](https://github.com/j-wilson/bot7/blob/master/examples/autoML.lua)) which takes a data file as input and creates a simple neural network using the optimized hyperparams.  

The file data/boston_automl.t7 in the data folder of this repo contains the data in the format expected by this module.
The format is detailed in the header of the autoML.lua file.

```
  th autoML.lua -data data/boston_automl.t7
```

### Hyperparams

The following are the best values I found from running manual tests which gave an MSE (Test Data) of 10 and use three hyperparams instead of five used in the tests.

```lua
  optimState = {
    learningRate = 1e-1,
    momentum = 0.4,
    learningRateDecay = 1e-4
  }
  opt.batch_size = 500
  opt.epochs = 20000
```
