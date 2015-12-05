--[[

  Implementation of the Attentive Reader described here:
  

Options:
  * input_dim   dimension of input vectors (required)
  * output_dim  dimension of output vectors (required)
  * cont_dim    dimension of controller state
  * cont_layers number of controller layers
  * shift_range allowed range for shifting read/write weights
  * write_heads number of write heads
  * read_heads  number of read heads
--]]

local ATTN, parent = torch.class('nn.ATTN', 'nn.Module')

function ATTN:__init(config)
  self.input_dim = config.input_dim or error('must have been assigned values. Recheck with rc-task.lua')
  self.output_dim  = config.output_dim  or error('config.output_dim must be specified')
  self.cont_dim    = config.cont_dim    or error('must have been assigned values. Recheck with rc-task.lua')
  self.cont_layers = config.cont_layers or error('must have been assigned values. Recheck with rc-task.lua')
  self.m_dim = config.m_dim or error('must have been assigned values. Recheck with rc-task.lua')
  self.g_dim = config.g_dim or error('must have been assigned values. Recheck with rc-task.lua')  -- column size of W(a), or length of g(d,q)
  self.dropout_rate = config.dropout or error('must have been assigned values. Recheck with rc-task.lua')

  -- lstm cells d and q don't share cells
  self.dfcells = {}
  self.dbcells = {}
  self.qfcells = {}
  self.qbcells = {}

  -- d and q shares parameters
  self.init_lstm_cell   = self:new_init_lstm_cell() --different weights.
  self.init_blstm_cell  = self:new_init_lstm_cell()
  self.master_fcell = self:new_fcell()
  self.master_bcell = self:new_bcell()

  ---attn
  self.ydt2stprime_cells ={}  -- put tdt2stprime cells
  self.master_ydt2stprime_cell = self:new_ydt2stprime_cell()  --recurse
  self.st2output_cell   = self:new_st2output_cell()  --single

end

--- initializes self.gradInput.
-- Need to be expanded further.XXXXXXXXXXXXXXXXXXXXXXXXX
function ATTN:init_lstm_grad_inputs()

  local m_gradInput, c_gradInput
  if self.cont_layers == 1 then
    m_gradInput = torch.zeros(self.cont_dim)
    c_gradInput = torch.zeros(self.cont_dim)
  else
    m_gradInput, c_gradInput = {}, {}
    for i = 1, self.cont_layers do
      m_gradInput[i] = torch.zeros(self.cont_dim)
      c_gradInput[i] = torch.zeros(self.cont_dim)
    end
  end

  self.gradInput = {
    torch.zeros(self.input_dim), -- input
    m_gradInput,
    c_gradInput
  }
end
function ATTN:init_blstm_grad_inputs()

  local m_gradInput, c_gradInput
  if self.cont_layers == 1 then
    m_gradInput = torch.zeros(self.cont_dim)
    c_gradInput = torch.zeros(self.cont_dim)
  else
    m_gradInput, c_gradInput = {}, {}
    for i = 1, self.cont_layers do
      m_gradInput[i] = torch.zeros(self.cont_dim)
      c_gradInput[i] = torch.zeros(self.cont_dim)
    end
  end

  self.bgradInput = {
    torch.zeros(self.input_dim), -- input
    m_gradInput,
    c_gradInput
  }
end

-- The initialization module initializes the state of NTM memory,
-- read/write weights, and the state of the LSTM controller.
function ATTN:new_init_lstm_cell()
  local dummy = nn.Identity()() -- always zero
  local output_init = nn.Tanh()(nn.Linear(1, self.input_dim)(dummy))

  -- controller state
  local m_init, c_init = {}, {}
  for i = 1, self.cont_layers do
    m_init[i] = nn.Tanh()(nn.Linear(1, self.cont_dim)(dummy))
    c_init[i] = nn.Tanh()(nn.Linear(1, self.cont_dim)(dummy))
  end

  -- wrap tables as nngraph nodes
  m_init = nn.Identity()(m_init)
  c_init = nn.Identity()(c_init)

  local inits = {
    output_init, m_init, c_init
  }
  return nn.gModule({dummy}, inits)
end

-- Create a new lstm forward cell. Each cell shares the parameters of the "master" cell
-- and stores the outputs of each iteration of forward propagation.
function ATTN:new_fcell()
  -- input to the network
  local input = nn.Identity()()

  -- LSTM controller output
  local mtable_p = nn.Identity()()
  local ctable_p = nn.Identity()()

  -- output and hidden states of the controller module
  local mtable, ctable = self:new_lstm_module(input, mtable_p, ctable_p)
  local m = (self.cont_layers == 1) and mtable
    or nn.SelectTable(self.cont_layers)(mtable)
  local output = nn.Identity()(m)  --TODO: add complicated output function

  local inputs = {input, mtable_p, ctable_p}
  local outputs = {output, mtable, ctable}

  local cell = nn.gModule(inputs, outputs)
  if self.master_fcell ~= nil then
    share_params(cell, self.master_fcell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

-- Create a new NTM backward cell. Each cell shares the parameters of the "master" cell
-- and stores the outputs of each iteration of forward propagation.
function ATTN:new_bcell()
  -- input to the network
  local input = nn.Identity()()

  -- LSTM controller output
  local mtable_p = nn.Identity()()
  local ctable_p = nn.Identity()()

  -- output and hidden states of the controller module
  local mtable, ctable = self:new_lstm_module(input, mtable_p, ctable_p)
  local m = (self.cont_layers == 1) and mtable
    or nn.SelectTable(self.cont_layers)(mtable)
  local output = nn.Identity()(m) -- TODO: add complicated output function

  local inputs = {input, mtable_p, ctable_p}
  local outputs = {output, mtable, ctable}

  local cell = nn.gModule(inputs, outputs)
  if self.master_bcell ~= nil then
    share_params(cell, self.master_bcell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

function ATTN:new_lstm_module(input, mtable_p, ctable_p)

  -- multilayer LSTM
  local mtable, ctable = {}, {}
  for layer = 1, self.cont_layers do
    local new_gate, m_p, c_p
    if self.cont_layers == 1 then
      m_p = mtable_p
      c_p = ctable_p
    else
      m_p = nn.SelectTable(layer)(mtable_p)
      c_p = nn.SelectTable(layer)(ctable_p)
    end

    if layer == 1 then
      new_gate = function()
        local in_modules = {
          nn.Dropout(self.dropout_rate)(nn.Linear(self.input_dim, self.cont_dim)(input)),
          nn.Linear(self.cont_dim, self.cont_dim)(m_p)
        }
        return nn.CAddTable()(in_modules)
      end
    else
      new_gate = function()
        return nn.CAddTable(){
          nn.Linear(self.cont_dim, self.cont_dim)(mtable[layer - 1]),
          nn.Linear(self.cont_dim, self.cont_dim)(m_p)
        }
      end
    end

    -- input, forget, and output gates
    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local o = nn.Sigmoid()(new_gate())
    local update = nn.Tanh()(new_gate())

    -- update the state of the LSTM cell
    ctable[layer] = nn.CAddTable(){
      nn.CMulTable(){f, c_p},
      nn.CMulTable(){i, update}
    }

    mtable[layer] = nn.CMulTable(){o, nn.Tanh()(ctable[layer])}
  end

  mtable = nn.Identity()(mtable)
  ctable = nn.Identity()(ctable)
  return mtable, ctable
end

--- lstm output -> encoding
-- doc: concat every state -- q: concat 1st and last state
function ATTN:new_encode_transform_cell(qsize)

  --doc
  local fwd_d = nn.Identity()()
  local bwd_d = nn.Identity()()
  local  yd = nn.JoinTable(2,2){fwd_d,bwd_d}
  --question
  local fwd_q = nn.Identity()()
  local bwd_q = nn.Identity()()
  local fwdSelect=nn.SelectTable(qsize)(nn.SplitTable(1)(fwd_q))
  local bwdSelect=nn.SelectTable(qsize)(nn.SplitTable(1)(bwd_q))
  local u = nn.JoinTable(1){fwdSelect,bwdSelect}

  return nn.gModule({fwd_d,bwd_d,fwd_q,bwd_q},{yd,u})
end

---------------------------------------------  Attention --------------------------------------------------

--recursive
function ATTN:new_ydt2stprime_cell()
  local ydt = nn.Identity()()
  local u   = nn.Identity()()
  local ydtemb = nn.Linear(2*self.cont_dim,self.m_dim)(ydt)
  local uemb    = nn.Linear(2*self.cont_dim,self.m_dim)(u)
  local mt = nn.Tanh()(nn.CAddTable(){ydtemb,uemb})
  local stprime = nn.Linear(self.m_dim,1)(mt)
  local ins = {ydt,u}
  local outs = {stprime}
  local cell = nn.gModule(ins,outs)
  if self.master_attn_cell ~= nil then
    share_params(cell, self.master_ydt2stprime_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end
-- single
function ATTN:new_st2output_cell()
  local stprime = nn.Identity()() --- aggregated st outputs
  local yd = nn.Identity()()
  local u = nn.Identity()()
  local sts_softmax = nn.SoftMax()(stprime)
  local vsts_softmax = nn.View()(sts_softmax)
  local r = nn.myMixtureTable(){vsts_softmax, yd}
  local gar = nn.Tanh()( nn.CAddTable(){nn.Linear(2*self.cont_dim,self.g_dim)(r), nn.Linear(2*self.cont_dim ,self.g_dim)(u)} )
  local padq = nn.LogSoftMax()(nn.Dropout(self.dropout_rate)(nn.Linear(self.g_dim,self.output_dim)(gar)))

  local ins = {yd,u,stprime}
  local outs = {padq}
  return nn.gModule(ins,outs)
end

-- alignment between question and each token in document
function ATTN:attend(yd,u)
  --  print(yd:size())
  --  print(u:size())
  local ydLen = yd:size(1)
  self.stprimes = torch.zeros(ydLen)
  for i=1, ydLen do
    if self.ydt2stprime_cells[i] ==nil then
      self.ydt2stprime_cells[i] = self:new_ydt2stprime_cell()
    end
    self.stprimes[i] = self.ydt2stprime_cells[i]:forward({yd[i],u})
  end
  --  print(self.stprimes:size())
  local output = self.st2output_cell:forward({yd,u,self.stprimes})
  return output
end

function ATTN:detend(yd,u,gradOutput)
  local ydLen = yd:size(1)
  local gyd = torch.zeros(yd:size()):fill(0)
  local gu  = torch.zeros(u:size()):fill(0)
--  print('--------------------In detend:--------------------')
  local gyd1,gu1,gstprimes = unpack( self.st2output_cell:backward({yd,u,self.stprimes}, gradOutput ) )
--  print('gyd from output module') print(gyd1:mean() )
--  print('gu from output module') print(gu1:mean() )

  for i=ydLen,1,-1 do
    local gydt, gu_ = unpack ( self.ydt2stprime_cells[i]:backward({yd[i],u},torch.Tensor{gstprimes[i]})  )
    gyd[i] = gydt
    gu:add(gu_)
  end
--  print('gyd from accumulated alignment') print(gyd:mean())
--  print('gu  from accumulated alignment') print(gu:mean())

  gyd:add(gyd1)
  gu:add(gu1)
  
  return gyd, gu
end
---------------------------------- cells end -------------------------------------------------------

-- generate output from a chain of hidden states
-- input: sentence (sentLen * tokenSize)
-- output: (sentLen * self.cont_dim)
function ATTN:encode(input, isdoc, isfwd)
  local cells =nil
  if isdoc then
    if isfwd then cells = self.dfcells else cells = self.dbcells end
  else
    if isfwd then cells = self.qfcells else cells = self.qbcells end
  end
  local newcell = function()
    if isfwd then return self:new_fcell() else return self:new_bcell() end
  end
  local init_cell = function()
    if isfwd then return self.init_lstm_cell else return self.init_blstm_cell end
  end
  local size = input:size(1)
  local fwdoutputs = torch.zeros(size,self.cont_dim)
  local prev_out
  for i = 1, size do
    if cells[i] == nil then
      cells[i] = newcell()
    end
    if i==1 then
      prev_out = init_cell():forward(torch.Tensor{0})
    else
      prev_out = cells[i-1].output
    end
    local cur_in = {input[i]}
    for i = 2, #prev_out do
      cur_in[i] = prev_out[i]
    end
    fwdoutputs[i]:copy( cells[i]:forward(cur_in)[1])
  end
  return fwdoutputs
end

-- grad = gradient of output.
function ATTN:decode(input,grad,isdoc,isfwd)
  local cells = nil
  if isfwd then
    if isdoc then cells = self.dfcells else cells = self.qfcells end
  else
    if isdoc then cells = self.dbcells else cells = self.qbcells end
  end

  if isfwd then self:init_lstm_grad_inputs() else self:init_blstm_grad_inputs() end-- set self.gradInput to zero start.

  local the_gradInput = nil
  if isfwd then the_gradInput = self.gradInput else the_gradInput = self.bgradInput end

  local initmodel = function()
    if isfwd then return self.init_lstm_cell else return self.init_blstm_cell end
  end
  local size = input:size(1)
  for k = size, 1, -1 do
    -- get gradients
    local cell = cells[k]
    local grad_outputs
    ---   LSTM looks like input ->[c,m]-> output.
    --   input's gradInput discarded.
    --   output's gradInput only have final state grad to bp thru.
    if k==size then
      if isdoc then grad_outputs = {grad[k]}  else grad_outputs = {grad[k]} end
    else
      if isdoc then grad_outputs = {grad[k]}  else grad_outputs = {torch.zeros(self.cont_dim)} end --TODO: confirm or grad[k]?
    end
    for i = 2, #the_gradInput do
      grad_outputs[i] = the_gradInput[i]
    end
    -- get inputs
    local prev_outputs
    if k == 1 then
      prev_outputs = initmodel():forward(torch.Tensor{0})
    else
      prev_outputs = cells[k - 1].output
    end
    local inputs = {input[k]}
    for i = 2, #prev_outputs do
      inputs[i] = prev_outputs[i]
    end
    --bp
    the_gradInput = cell:backward(inputs, grad_outputs)

    if k == 1 then
      initmodel():backward(torch.Tensor{0}, the_gradInput)
    end
  end
end
--- Forward Backward Functions:
-- input: {
--            Document Tensor (SentenceLen x TokenSize)
--            Question Tensor (QuestionLen x TokenSize)
--         }
function ATTN:forward(input)
  self.dsize = input[1]:size(1)
  self.qsize = input[2]:size(1)
  self.fwd_d = self:encode(input[1],true,true)
  self.bwd_d = self:encode( nn.RowReverse():forward(input[1]),true,false)
  self.fwd_q = self:encode(input[2],false,true)
  self.bwd_q = self:encode(nn.RowReverse():forward(input[2]),false,false)
  self.encode_transform_cell = self:new_encode_transform_cell(self.qsize) --transform_cell does not have parameters.
  self.yd, self.u = unpack( self.encode_transform_cell:forward({self.fwd_d,self.bwd_d,self.fwd_q,self.bwd_q}) )
  --  print('yd size is') print(self.yd:size())
--  print('Forward:  self.yd and self.u is ' .. self.yd:mean() .. ' ' .. self.u:mean())
  self.output = self:attend(self.yd,self.u)
  return self.output
end
function ATTN:backward(input, grad_output)

--  print('grad for attention is ' .. grad_output:mean())
  local gyd,gu =  self:detend(self.yd,self.u,grad_output)
--  print('grad for transform_cell is') print('gyd ' ) print( gyd:mean() ) print(' gu ') print( gu:mean() )
  local gfwd_d, gbwd_d,gfwd_q,gbwd_q = unpack( self.encode_transform_cell:backward({self.fwd_d,self.bwd_d,self.fwd_q,self.bwd_q}, {gyd,gu}) )
--  print('grad for encoders: ')
--  print('gfwd_d ') print(gfwd_d:mean())print('gbwd_d ') print(gbwd_d:mean())
--  print('gfwd_q') print(gfwd_q:mean()) print('gbwd_q') print(gbwd_q:mean())
  self:decode(nn.RowReverse():forward(input[2]), gbwd_q,false,false)
  self:decode(input[2],gfwd_q,false,true)
  self:decode(nn.RowReverse():forward(input[1]), gbwd_d,true,false)
  self:decode(input[1],gfwd_d,true,true)
  return self.gradInput -- dummy.
end

function ATTN:parameters()
  local p,g = self.init_lstm_cell:parameters()
  local p1,g1 = self.init_blstm_cell:parameters()
  local p2,g2 = self.master_fcell:parameters()
  local p3,g3 = self.master_bcell:parameters()
  local p4,g4 = self.master_ydt2stprime_cell:parameters()
  local p5,g5 = self.st2output_cell:parameters()

  tablex.insertvalues(p, p1)
  tablex.insertvalues(g, g1)
  tablex.insertvalues(p, p2)
  tablex.insertvalues(g, g2)
  tablex.insertvalues(p, p3)
  tablex.insertvalues(g, g3)
  tablex.insertvalues(p, p4)
  tablex.insertvalues(g, g4)
  tablex.insertvalues(p, p5)
  tablex.insertvalues(g, g5)
  return p, g
end

function ATTN:zeroGradParameters()
  self.init_lstm_cell:zeroGradParameters()
  self.init_blstm_cell:zeroGradParameters()
  self.master_fcell:zeroGradParameters()
  self.master_bcell:zeroGradParameters()
  self.master_ydt2stprime_cell:zeroGradParameters()
  self.st2output_cell:zeroGradParameters()
end
