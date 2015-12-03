--[[

  Training attentive reader

  The current version seems to work, giving good output after 5000 iterations
  or so. Proper initialization of the read/write weights seems to be crucial
  here.

--]]

require('../init')
require('optim')
require('sys')
require('Data')

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('--cont_dim', 32, '')
cmd:option('--cont_layers', 1, '')
cmd:option('--m_dim',32,'')
cmd:option('--g_dim',32,'')
cmd:option('--train','','must specify train file path')
cmd:option('--valid', '', 'must specify validation file path')
cmd:option('--testonly',false,'')
cmd:option('--loadmodel','none','')
cmd:option('--full_vocab_output',false,'')
cmd:option('--eval_interval',30000,'')
g_params = cmd:parse(arg or {})

print(g_params)

torch.manualSeed(0)
math.randomseed(0)

---------------------------- read data --------------------------------------------
--
--
local data = Data()
local desc, q, a = data:read(g_params.train,true,g_params.full_vocab_output)  --- not only reads data, but also forms dictionaries.
local tdesc, tq, ta = data:read(g_params.valid,false,g_params.full_vocab_output)   --- use dictionaries from previous state to get validation data.
collectgarbage()
print( 'vocabulary size for training data is ' .. #data:getVocab())

-- note: permutation of training set is done during training.
-- Never permute test set!

local l_output_dim = #data:getEVocab()
if g_params.full_vocab_output then l_output_dim = #data:getVocab()+2 end

local config = {
  input_dim   =   #data:getVocab()+1,
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
  local nd,nq,na = data:permute_an_example(desc,q,a,ind)
  return nd,nq,na
    --return desc[ind],q[ind],a[ind]
end


local criteria = nn.ClassNLLCriterion()

local desc_matrix = torch.Tensor()
local question_matrix = torch.Tensor()

function forward_backward(model, desc, question, answer, print_flag)

  -- wwww  <s> wwwwwwwww <s> wwwwwwwwwww <s>
  local descLen = 0
  for i = 1, desc:size(1) do
    for j = 1, desc:size(2) do
      if desc[i][j]~=0 then
        descLen = descLen+1
      end
    end
    descLen = descLen+1
  end
  desc_matrix:resize(descLen,config.input_dim):zero()
  local count = 0
  for i = 1, desc:size(1) do
    for j = 1, desc:size(2) do
      if desc[i][j]~=0 then
        count = count+1
        desc_matrix[count][desc[i][j]] =1
      end
    end
    count = count+1
    desc_matrix[count][config.input_dim]=1
  end

  question_matrix:resize(question:size(1),config.input_dim):zero()
  for i=1,question:size(1) do
    question_matrix[i][question[i]]=1
  end

  -- present targets
  local output = model:forward({desc_matrix, question_matrix}) -- output is n hot encoding of dictionary size.
  local  loss = criteria:forward(output, answer)

  model:backward({desc_matrix,question_matrx},loss)

  return output, loss
end


function evaluate(model,descs,questions,answers)
  local vocab = data:getVocab()
  if not g_params.full_vocab_output then vocab = data:getEVocab() end
  print('vocab size is ' .. #vocab)
  local correct = 0
  for i=1, #descs do
    print('evaluating ' .. i .. 'th example')
    local shortlist = get_entity_shortlist(descs[i],vocab)

    --    print('print shortlist paris')
    --    for k,v in pairs(shortlist) do
    --        print (k .. ' ' .. v)
    --    end
    local output,loss,criteria = forward(model,descs[i],questions[i],answers[i], false)
    backward(model,descs[i],questions[i],answers[i],output, criteria, false)
    local p = output_argmax(output,shortlist)
    if p == answers[i] then correct = correct +1 end
    print (' answer is ' .. answers[i] .. ' , prediction is ' .. p)
  end
  print('accuracy is ' .. (correct/#descs))
end

function get_entity_shortlist(desc,vocab)
  local shortlist = {}
  for i = 1, desc:size(1) do
    for j = 1, desc:size(2) do
      if desc[i][j]~=0 then
        if data:getEVocab()[data:getiVocab()[ desc[i][j] ] ] then
          if g_params.full_vocab_output then
            shortlist[desc[i][j]]=1
          else
            shortlist[ data:getEVocab()[ data:getiVocab()[ desc[i][j] ]  ]   ] =1
          end
        end
      end
    end
  end
  return shortlist
end

function output_argmax(output,shortlist)
  local maxind=0
  local maxval= -math.huge
  for i=1, output:size(1) do
    if output[i] > maxval and shortlist[i] then
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
local params, grads = ntmmodel:getParameters()

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

if g_params.testonly then
  if g_params.loadmodel ~= 'none' then
    params:copy(torch.load(g_params.loadmodel))
    evaluate(ntmmodel, tdesc,tq,ta)
  else
    error("should load model")
  end
else
  ------------------------------- train and test -------------------------------------------
  -- train
  for iter = 1, num_iters do
    local eval_flag = iter % eval_interval ==0

    local feval = function(x)
      print(string.rep('-', 80))
      print('iter = ' .. iter)
      print('learn rate = ' .. rmsprop_state.learningRate)
      print('momentum = ' .. rmsprop_state.momentum)
      print('decay = ' .. rmsprop_state.decay)
      printf('t = %.1fs\n', sys.clock() - start)

      local loss = 0
      grads:zero()

      local dt ,qt, at = generate_example(data,desc,q,a)
      --   print("generated example")
      local output, loss = forward_backward(ntmmodel, dt, qt, at, false)
      --   print("forward done")
      --   print("backward done")

      -- clip gradients
      grads:clamp(-10, 10)
      print('max grad = ' .. grads:max())
      print('min grad = ' .. grads:min())
      print('loss = ' .. loss)
      return loss, grads
    end

    --optim.adagrad(feval, params, adagrad_state)
    ntm.rmsprop(feval, params, rmsprop_state)

    if eval_flag then
      evaluate(ntmmodel, tdesc,tq,ta)
    end
    --  collectgarbage()
  end
end
