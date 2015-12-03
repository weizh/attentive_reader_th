require('torch')
require('nn')
require('nngraph')


include('rmsprop.lua')
include('nn/LinearNoBias.lua')
include('nn/RowReverse.lua')
include('nn/myMixtureTable.lua')
include('attn.lua')

function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end
