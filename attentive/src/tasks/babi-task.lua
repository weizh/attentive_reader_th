--[[

  Training attentive reader

  http://arxiv.org/pdf/1506.03340.pdf

--]]

require('../init')
require('optim')
require('sys')
require('babi')

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('--cont_dim', 100, '')
cmd:option('--cont_layers', 1, '')
cmd:option('--m_dim',100,'')
cmd:option('--g_dim',100,'')
cmd:option('--train','','must specify train file path')
cmd:option('--valid', '', 'must specify validation file path')
cmd:option('--dropout',0.00005,'')
cmd:option('--testonly',false,'')
cmd:option('--loadmodel','none','')
cmd:option('--full_vocab_output',false,'')
cmd:option('--eval_interval',1000,'')
g_params = cmd:parse(arg or {})

print(g_params)

torch.manualSeed(0)
math.randomseed(0)

---------------------------- read data --------------------------------------------
--
--
local data = BABI()
local desc, q, a = data:readFile(g_params.train,true,false)  
local tdesc, tq, ta = data:readFile(g_params.valid,false,false)   

collectgarbage()

-- note: permutation of training set is done during training.
-- Never permute test set!

local l_output_dim = #data:get_outVocab()

local config = {
  input_dim   =   #data:get_inVocab()+1,
  output_dim  =   l_output_dim,
  cont_dim    =   g_params.cont_dim,
  cont_layers =   g_params.cont_layers,
  m_dim =  g_params.m_dim,
  g_dim =  g_params.g_dim,
  dropout = g_params.dropout

}

--------------------------------start program -----------------------------------------------------------
function generate_example(data,desc,q,a)
  local ind = math.floor(torch.uniform(1, #desc+0.99))
  return desc[ind],q[ind],a[ind]
end


local criteria = nn.ClassNLLCriterion()

local desc_matrix = torch.Tensor()
local question_matrix = torch.Tensor()

function forward_backward(model, desc, question, answer, print_flag)

  -- wwww  <s> wwwwwwwww <s> wwwwwwwwwww <s>

  desc_matrix:resize(#desc,config.input_dim):zero()

  local count = 0
    for i,v in pairs(desc) do
      count = count+1
      if desc[i]~=0 then 
	desc_matrix[count][v]=1 
      end
    end
  question_matrix:resize(#question,config.input_dim):zero()
  for i,v in pairs(question) do
    question_matrix[i][v]=1
  end

--  print(' --------------------------- Running Forward ------------------------------ ')
  -- present targets
  local output = model:forward({desc_matrix, question_matrix}) -- output is n hot encoding of dictionary size.
--  print('output mean is ') print(output:mean())

--  print(' --------------------------- Running Backward ------------------------------ ')
  local  loss = criteria:forward(output, answer[1])
  local criout= criteria:backward(output,answer[1])
--  print('loss is') print(loss)

  model:backward( {desc_matrix,question_matrix} , criout )

  return output, loss
end


function evaluate(model,descs,questions,answers)
  local vocab = data:get_inVocab()
--  print('vocab size is ' .. #vocab)
  local correct = 0
    print('evaluating ..')

  for i=1, #descs do
--    print('evaluating ' .. i .. 'th example')

    --    print('print shortlist paris')
    --    for k,v in pairs(shortlist) do
    --        print (k .. ' ' .. v)
    --    end
    local output,loss,criteria = forward_backward(model,descs[i],questions[i],answers[i], false)
    local p = output_argmax(output)
    if p == answers[i][1] then correct = correct +1 end
--    print (' answer is ' .. answers[i][1] .. ' , prediction is ' .. p)
--    print (' running accuracy is ' .. (correct/i) )
  end
  print('Final accuracy is ' .. (correct/#descs))
end


function output_argmax(output)
  local maxind=0
  local maxval= -math.huge
  for i=1, output:size(1) do
    if output[i] > maxval then
      maxval = output[i]
      maxind = i
    end
  end
  return maxind
end


--------------------------------------------- main starts here ------------------------------------------------------------------------------------------
--
--
local attn = nn.ATTN(config)
local params, grads = attn:getParameters()

local num_iters = 1E10
local start = sys.clock()
local print_interval = 25
local eval_interval = g_params.eval_interval
print(string.rep('=', 80))
print("Attentive reader rc data task")
print('training up to ' .. num_iters .. ' iteration(s)')
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local rmsprop_state = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

local adagrad_state = {
  learningRate = 1e-3
}

if g_params.testonly then
  if g_params.loadmodel ~= 'none' then
    params:copy(torch.load(g_params.loadmodel))
    evaluate(attn, tdesc,tq,ta)
  else
    error("should load model")
  end
else
  ------------------------------- train and test -------------------------------------------
  -- train
  for iter = 1, num_iters do

    --    if iter >4 then break end
    local eval_flag = iter % eval_interval ==0
    local print_flag = iter%print_interval ==0
    local feval = function(x)

if print_flag then
      print(string.rep('-', 200))
      print('iter = ' .. iter)
      print('learn rate = ' .. rmsprop_state.learningRate)
      print('momentum = ' .. rmsprop_state.momentum)
      print('decay = ' .. rmsprop_state.decay)
      --      print('learn rate = ' .. adagrad_state.learningRate)
      print('t = ' ..  (sys.clock() - start) )
end
      local loss = 0
      grads:zero()

      local dt ,qt, at = generate_example(data,desc,q,a)
      --   print("generated example")
      local output, loss = forward_backward(attn, dt, qt, at, false)
      --   print("forward done")
      --   print("backward done")

      -- clip gradients
      grads:clamp(-10, 10)
if print_flag then
      print('max grad = ' .. grads:max())
      print('min grad = ' .. grads:min())
      print('loss = ' .. loss)
end
      return loss, grads
    end

    --    optim.adagrad(feval, params, adagrad_state)
    rmsprop(feval, params, rmsprop_state)

    if eval_flag then
      evaluate(attn, tdesc,tq,ta)
    end
    --  collectgarbage()
  end
end
