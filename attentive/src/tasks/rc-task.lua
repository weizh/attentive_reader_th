--[[

  Training a NTM to memorize input.

  The current version seems to work, giving good output after 5000 iterations
  or so. Proper initialization of the read/write weights seems to be crucial
  here.

--]]

require('../init')
require('./util')
require('optim')
require('sys')
require('Data')
require('../rmsprop')
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('--cont_dim', 100, '')
cmd:option('--cont_layers', 1, '')
cmd:option('--train','','must specify train file path')
cmd:option('--valid', '', 'must specify validation file path')
cmd:option('--testonly',false,'')
cmd:option('--loadmodel','none','')
cmd:option('--eval_interval',30000,'')

g_params = cmd:parse(arg or {})

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

--Attentive reader configuration
local config = {
  input_dim = #data:getVocab()+2,
  output_dim = l_output_dim,
  cont_dim = g_params.cont_dim,
  cont_layers =g_params.cont_layers,
}

function generate_example(data,desc,q,a)
  local ind = math.floor(torch.uniform(1, #desc+0.99))
  local nd,nq,na = data:permute_an_example(desc,q,a,ind)
  return nd,nq,na
end

--------------------------------start program -----------------------------------------------------------
--start_symbol = torch.zeros(config.input_dim)
--start_symbol[config.input_dim]=1
--query_symbol = torch.zeros(config.input_dim)
--query_symbol[config.input_dim-1]=1
--zero_symbol = torch.zeros(config.input_dim)

function forward(flstm, blstm, attn, desc, question, answer, print_flag)

  local dsize = get_desc_size(desc)
  local d_f_output = encode_d(flstm, desc, dsize,config.cont_dim,false)
  local d_b_output = encode_d(blstm, desc, dsize,config.cont_dim,false)

  local q_f_output= encode_q(flstm, question, config.cont_dim,false)
  local q_d_output = encode_q(blstm, question,config.cont_dim,false)

  local logSoftMax   = attend(attn, d_f_output, d_b_output, q_f_output, q_d_output)

  local criteria = nn.ClassNLLCritrerion()

  local loss = criteria:forward(logSoftMax, answer)

  return logSoftMax, loss, criteria
end

function get_desc_size(desc)
  local size =0
  for i = 1, desc:size(1) do
    for j = 1, desc:size(2) do
      if desc[i][j]~=0 then
        size = size+1
      end
    end
  end
  return size
end

-- Encoded ensures that no zero peddings between sentences are there.
function encode_d( lstm, desc, size, rnnsize, flag)
  local w = torch.zeros(config.input_dim)
  local output=torch.DoubleTensor(size, rnnsize)
  local index = 1
  for i = 1, desc:size(1) do
    for j = 1, desc:size(2) do
      if desc[i][j]~=0 then
        w[desc[i][j]]=1
        outputs[index] = model:forward(w)
        index = index+1
        w[desc[i][j]]=0
      end
    end
  end
  return output
end

function encode_q(lstm, question,flag)
  local w = torch.zeros(config.input_dim)
  local output = torch.DoubleTensor(question:size(1))
  for i=1,question:size(1) do
    w[question[i]]=1
    output[i]=model:forward(w)
    w[question[i]]=0
  end
  return output
end

function attend(attn, dfout,dbout,qfout,qbout)
  local y_d = fbconcat_desc(dfout, dbout)
  local u   = htconcat_question(qfout, qbout)
  return attn:forward(y_d,u)
end

--  local target = torch.zeros(config.output_dim)
--  -- present start symbol
--  model:forward(start_symbol)
--
--  -- present inputs
--
--
--  -- present targets
--  model:forward(query_symbol)
--  local output = model:forward(zero_symbol) -- output is n hot encoding of dictionary size.
--
--    criteria = nn.ClassNLLCriterion()
--
--  target[answer]=1
--  local loss
--    loss = criteria:forward(output, answer)
--  target[answer]=0
--
--  --  if print_flag then print_read_max(model) end
--
--  return output, loss, criteria
--end

function attend(model, desc, question, answer, print_flag)
  local d_f_output, d_f_loss, q_f_criteria = encode(f_lstm_model, dt, false)
  local d_b_output, d_b_loss, q_b_criteria = encode(b_lstm_model, dt, false)

  local q_f_output, q_f_loss, q_f_criteria = encode(f_lstm_model, qt, false)
  local q_d_output, q_d_loss, q_d_criteria = encode(b_lstm_model, qt, false)

  local y_d = fw_bw_concat

  local atte
  return output, loss, criteria
end


function backward(model, desc, question, answer, output,criteria, train)
  local w = torch.zeros(config.input_dim)
  local target = torch.zeros(config.output_dim)
  local zeros = torch.zeros(config.output_dim)

  --  print('backward readout symbol')
  if train then
    target[answer] = 1
    if config.output_trans == 1 then
      model:backward(zero_symbol, criteria:backward(output,target):mul(config.input_dim))
    elseif config.output_trans == 0 then
      model:backward(zero_symbol, criteria:backward(output,answer))
    else
      error('specify output transformation first. -- thrown in backward.')
    end
    target[answer]=0
  else
    model:backward(zero_symbol,zeros)
  end
  --  print('backward readout symbol')
  model:backward(query_symbol,zeros)

  --  print('backward question symbol')
  for j = question:size(1),1, -1 do
    w[question[j]]=1
    model:backward(w,zeros)
    w[question[j]]=0
  end

  --  print('backward query symbol')
  model:backward(query_symbol,zeros)

  --  print('backward description symbol')
  for i = desc:size(1), 1, -1 do
    for j = desc:size(2),1,-1 do
      if desc[i][j]~=0 then
        w[desc[i][j]]=1
        model:backward(w,zeros)
        w[desc[i][j]]=0
      end
    end
  end

  --  print('backward start symbol')
  model:backward(start_symbol, zeros)
  --  print('backward model. NTM cell left: ' .. model:get_depth())
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
local f_lstm_model = LSTM(config)
local b_lstm_model = lSTM(config)

local attn_model = ATTN(config)

local params, grads = attn_mmodel:getParameters()

local num_iters = 1E10
local start = sys.clock()
local print_interval = 25
local eval_interval = g_params.eval_interval
print(string.rep('=', 80))
print("NTM rc data task")
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
    local print_flag = iter% print_interval ==0

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
      local output, loss, criteria = forward(f_lstm_model, b_lstm_model, attn_model, dt,qt,at,print_flag)
      --   print("forward done")
      backward(f_lstm_model, b_lstm_model, attn_model, dt, qt, at, output,criteria, true)
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
