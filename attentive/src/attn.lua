--[[

  Implementation of the Neural Turing Machine described here:

  http://arxiv.org/pdf/1410.5401v2.pdf

  Variable names take after the notation in the paper. Identifiers with "r"
  appended indicate read-head variables, and likewise for those with "w" appended.

  The NTM take a configuration table at initialization time with the following
  options:

  * input_dim   dimension of input vectors (required)
  * output_dim  dimension of output vectors (required)
  * mem_rows    number of rows of memory
  * mem_cols    number of columns of memory
  * cont_dim    dimension of controller state
  * cont_layers number of controller layers
  * shift_range allowed range for shifting read/write weights
  * write_heads number of write heads
  * read_heads  number of read heads

--]]

local attn, parent = torch.class('ATTN', 'nn.Module')


function attn:__init(config)
  self.input_dim   = config.input_dim   or error('config.input_dim must be specified')
  self.output_dim  = config.output_dim  or error('config.output_dim must be specified')
  self.cont_dim    = config.cont_dim    or 100
  self.cont_layers = config.cont_layers or 1
  self.dropout = config.dropout or 0.5
  self.depth = 0
  self.cells = {}
  
  self.master_cell = self:new_cell()
  
  self.init_module = self:new_init_module()
  self:init_grad_inputs()

end

function ATTN:new_attention()
  
end

function LSTM:init_grad_inputs()

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

-- The initialization module initializes the state of NTM memory,
-- read/write weights, and the state of the LSTM controller.
function LSTM:new_init_module()
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

-- Create a new NTM cell. Each cell shares the parameters of the "master" cell
-- and stores the outputs of each iteration of forward propagation.
function LSTM:new_cell()
  -- input to the network
  local input = nn.Identity()()

  -- LSTM controller output
  local mtable_p = nn.Identity()()
  local ctable_p = nn.Identity()()

  -- output and hidden states of the controller module
  local mtable, ctable = self:new_controller_module(input, mtable_p, ctable_p)
  local m = (self.cont_layers == 1) and mtable 
    or nn.SelectTable(self.cont_layers)(mtable)
  local output = self:new_output_module(m)

  local inputs = {input, mtable_p, ctable_p}
  local outputs = {output, mtable, ctable}

  local cell = nn.gModule(inputs, outputs)
  if self.master_cell ~= nil then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end  
  return cell
end

-- Create a new LSTM controller
function LSTM:new_controller_module(input, mtable_p, ctable_p)
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
----------------------- apply dropout? -----------------------------------------------------------------------
          nn.Dropout(self.dropout)(nn.Linear(self.input_dim, self.cont_dim)(input)),
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

-----------------------------------------------------------------------------------------------------------------------------------------------------
-- Create an output module, e.g. to output binary strings.
function LSTM:new_output_module(m)
-- return nn.LogSoftMax()(nn.Sigmoid()(nn.Linear(self.cont_dim, self.output_dim)(m)))
--   return nn.Sigmoid()(nn.Dropout(self.dropout)(nn.Linear(self.cont_dim, self.output_dim)(m)))
  return m
end

-- Forward propagate one time step. The outputs of previous time steps are 
-- cached for backpropagation.
function LSTM:forward(input)
  self.depth = self.depth + 1
  local cell = self.cells[self.depth]
  if cell == nil then
    cell = self:new_cell()
    self.cells[self.depth] = cell
  end
  
  local prev_outputs
  if self.depth == 1 then
    prev_outputs = self.init_module:forward(torch.Tensor{0})
  else
    prev_outputs = self.cells[self.depth - 1].output
  end

  -- get inputs
  local inputs = {input}
  for i = 2, #prev_outputs do
    inputs[i] = prev_outputs[i]
  end
  local outputs = cell:forward(inputs)
  self.output = outputs[1]
  return self.output
end

-- Backward propagate one time step. Throws an error if called more times than
-- forward has been called.
function LSTM:backward(input, grad_output)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end
  local cell = self.cells[self.depth]
  local grad_outputs = {grad_output}
  for i = 2, #self.gradInput do
    grad_outputs[i] = self.gradInput[i]
  end

  -- get inputs
  local prev_outputs
  if self.depth == 1 then
    prev_outputs = self.init_module:forward(torch.Tensor{0})
  else
    prev_outputs = self.cells[self.depth - 1].output     -------------------------------------
  end
  local inputs = {input}
  for i = 2, #prev_outputs do
    inputs[i] = prev_outputs[i]
  end

  self.gradInput = cell:backward(inputs, grad_outputs)
  self.depth = self.depth - 1
  if self.depth == 0 then
    self.init_module:backward(torch.Tensor{0}, self.gradInput)
    for i = 1, #self.gradInput do
      local gradInput = self.gradInput[i]
      if type(gradInput) == 'table' then
        for _, t in pairs(gradInput) do t:zero() end
      else
        self.gradInput[i]:zero()
      end
    end
  end
  return self.gradInput
end


function LSTM:parameters()
  local p, g = self.master_cell:parameters()
  local pi, gi = self.init_module:parameters()
  tablex.insertvalues(p, pi)
  tablex.insertvalues(g, gi)
  return p, g
end

function LSTM:forget()
  self.depth = 0
  self:zeroGradParameters()
  for i = 1, #self.gradInput do
    self.gradInput[i]:zero()
  end
end

function LSTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
  self.init_module:zeroGradParameters()
end
