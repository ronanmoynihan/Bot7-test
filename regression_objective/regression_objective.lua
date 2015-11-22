require 'torch'
require 'optim'
require 'math'
require 'nn'
local data_loader = require 'regression_objective.data'
local train = require 'regression_objective.train'
local test = require 'regression_objective.test_model'

local data = nil

local regression_objective = function(X, test_hypers)
	
	if (X:dim() == 1 or X:size(1) == X:nElement()) then
    	X = X:reshape(1, X:nElement())
  	end
  	assert(X:size(2) == 5)

	-- The Boston housing data has been converted to torch.
	-- http://lib.stat.cmu.edu/datasets/boston
	local data_file = 'data/boston.t7'

	torch.manualSeed(4)

	local opt = {}

	opt.model_name = 'model'
	opt.optimization = 'sgd'

	-- NOTE: the code below changes the optimization algorithm used, and its settings
	local optimState       -- stores a lua table with the optimization algorithm's settings, and state during iterations
	local optimMethod      -- stores a function corresponding to the optimization routine

	if opt.optimization == 'sgd' then
	  optimState = {
	  	learningRate = X[1][1],
	  	momentum = X[1][2],
	  	learningRateDecay = X[1][3],
	  	weightDecay = X[1][4],
	  	dropout = X[1][5]
	  }
	  opt.batch_size = 500
	  opt.epochs = 20000
	  optimMethod = optim.sgd
	else
	  error('Unknown optimizer')
	end

	local data_train_percentage = 70 
	data = data or data_loader.load_data(data_file, data_train_percentage)

	-- Train.
	local model, final_batch_loss, test_loss = train(opt,optimMethod,optimState, data, nn.MSECriterion())

	if test_hypers then
		test(data,model)
		print(string.format('\nBot7 Hyperparams on test data MSE: %d', test_loss))
	end	

	return final_batch_loss

end

return regression_objective




