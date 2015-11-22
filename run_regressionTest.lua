local bot7  = require('bot7')
local hyperparam = bot7.hyperparam
local regression_objective = require 'regression_objective.regression_objective'

---------------- Argument Parsing
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-bot',       'bo', 'specify which bot to use: {bo, rs}')
cmd:option('-nInitial',   2, 'specify number of initial candidates to sample at random')
cmd:option('-budget',     100,'specify budget (#nominees) for experiment')
cmd:option('-noisy',      false, 'specify observations as noisy')
cmd:option('-grid_size',  20000, 'specify size of candidate grid')
cmd:option('-grid_type',  'random', 'specify type for candidate grid')
cmd:option('-mins', '',   'specify minima for inputs (defaults to 0.0)')
cmd:option('-maxes',      '', 'specify maxima for inputs (defaults to 1.0)')
cmd:option('-score',      'ei', 'specify acquisition function to be used by bot; {ei, ucb}')

cmd:text()
opt = cmd:parse(arg or {})
opt.xDim = 5
opt.yDim = 1

---------------- Experiment Configuration
local expt = 
{
  xDim   = opt.xDim,
  yDim   = opt.yDim,
  budget = opt.budget,
}

expt.model = {noiseless = not opt.noisy} 
expt.grid  = {type = opt.grid_type, size = opt.grid_size}
expt.bot   = {type = opt.bot, nInitial = opt.nInitial}

-- Establish feasible hyperparameter ranges
if (opt.mins ~= '') then
  loadstring('expt.mins='..opt.mins)()
else
  expt.mins = torch.zeros(1, opt.xDim)
end

if (opt.maxes ~= '') then
  loadstring('opt.maxes='..opt.maxes)()
else
  expt.maxes = torch.ones(1, opt.xDim)
end

---- Choose acquistion function
expt['score'] = {}
if (opt.score == 'ei') then
  expt.score['type'] = 'expected_improvement'
elseif (opt.score == 'ucb') then
  expt.score['type'] = 'confidence_bound'
end

---- Set metatables
for key, val in pairs(expt) do
  if type(val) == 'table' then
    setmetatable(val, {__index = expt})
  end
end

------------------------------------------------
--                                run_regression
------------------------------------------------
function run_regressionTest(expt)
  local hypers = {}
  for k = 1, opt.xDim do
    hypers[k] = hyperparam('x'..k, 0, 1)
  end

  -------- Initialize bot
  bot = bot7.bots.bayesopt(regression_objective, hypers, expt)

  -------- Perform experiment
  bot:run_experiment()
  
end

run_regressionTest(expt)
regression_objective(bot.best.x,true)
print(string.format('\nbot.best.y (Training MSE): %d', bot.best.y[1][1]))