---
--This function swaps the rows of a matrx. It also works in batch.
--
--The rows are flipped in reverse order, such as a[{1+i] and a[{#a-i] are swaped.

local RowReverse, parent = torch.class('nn.RowReverse', 'nn.Module')

function RowReverse:__init()
  parent.__init(self)
  --  self.output
  --  self.gradInput
  self.temp = torch.Tensor()
  self.input_block = torch.Tensor()
end

function RowReverse:updateOutput(input)

  self.output:resizeAs(input)

  if input:dim() ==2 then
    local len = input:size(1)
    for i = 1, len do
      self.output[len-i+1]:copy(input[i])
    end
  elseif input:dim() ==3 then
    for iiter=1,input:size(1) do      
      self.input_block:resizeAs(input[iiter]):copy(input[iiter]) -- 2d copy of input
      local len = self.input_block:size(1)
      for i = 1, len do
        self.output[iiter][len-i+1]:copy(input[iiter][i])
      end
    end
  else
    error('The input into nn.ColReverse should never be vectors or tensor with >3 dims.')
  end
  return self.output
end

function RowReverse:updateGradInput(input, gradOutput)

  if input:dim() ~= gradOutput:dim() then
   error(' dimension does not match.')
  elseif input:dim()==2 then
    if input:size(1) ~=gradOutput:size(1) then error('GradOutput and input first dim does not match.') end
    if input:size(2) ~=gradOutput:size(2) then error('GradOutput and input second dim does not match.') end
  elseif input:dim()==3 then
    if input:size(1) ~=gradOutput:size(1) then error('GradOutput and input first dim does not match.') end
    if input:size(2) ~=gradOutput:size(2) then error('GradOutput and input second dim does not match.') end
    if input:size(3) ~=gradOutput:size(3) then error('GradOutput and input second dim does not match.') end
  else
    error('dimension is not 2 or 3. Illegal.')
  end
  
  self.gradInput:resizeAs(gradOutput)

  if gradOutput:dim() ==2 then
    local len = gradOutput:size(1)
    for i = 1, len do
      self.gradInput[len-i+1]:copy(gradOutput[i])
    end
  elseif gradOutput:dim() ==3 then
    for iiter=1,gradOutput:size(1) do      
      self.input_block:resizeAs(gradOutput[iiter]):copy(gradOutput[iiter]) -- 2d copy of input
      local len = self.input_block:size(1)
      for i = 1, len do
        self.gradInput[iiter][len-i+1]:copy(gradOutput[iiter][i])
      end
    end
  else
    error('The input into nn.ColReverse should never be vectors or tensor with >3 dims.')
  end
  return self.gradInput
  
end

function RowReverse:accGradParameters(input, gradOutput)
  
end

function RowReverse:reset()
end
