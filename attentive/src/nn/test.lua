require 'torch'
require 'nn'
require 'nngraph'

include('RowReverse.lua')

local a = torch.rand(3,4,5)

local b = nn.RowReverse():forward(a)

print(a)

print(b)