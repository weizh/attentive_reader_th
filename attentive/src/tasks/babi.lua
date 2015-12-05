local babi = torch.class('BABI')

local stringx = require('pl.stringx')
local file = require('pl.file')
local tds = require('tds')

function babi:__init()
  self.desc_vocab  = tds.hash()
  self.q_vocab     = tds.hash()
  self.desc        = {}
  self.qs          = {}
  self.ans         = {}
end

function babi:get_outVocab()
  return self.q_vocab
end

function babi:get_inVocab()
  return self.desc_vocab
end
function babi:gen_dict(fname)
  for line in io.lines(fname) do
    self:fill_dict(stringx.strip(line))
  end
end

function babi:fill_dict(line)
  local parts = stringx.split(line,'\t')
  if #parts>1 then
    local sp = stringx.strip(parts[2])
    if not self.q_vocab[sp] then
      self.q_vocab[sp] = #self.q_vocab+1
    end
  end

  local text = stringx.split(stringx.strip(parts[1]))
  for i=2,#text do
    if not self.desc_vocab[text[i]] then
      self.desc_vocab[text[i]]=#self.desc_vocab+1
    end
  end
end

function babi:readFile(fname,istrain, suponly)
  if istrain then
    self:gen_dict(fname)
    print(self.desc_vocab)
    print(self.q_vocab)
  end
  --  print(fname)
  local tokens
  local tlines = {}
  for line in io.lines(fname) do  --print(line)
    tokens = stringx.split(line, ' ')
    if tonumber(tokens[1]) ==1 and #tlines~=0 then -- a new instance
      tlines = {tokens}  -- reset tlines
    elseif string.find(tokens[#tokens],'\t') then
      self:process(tlines,tokens,suponly)
    else
      tlines[tonumber(tokens[1])] = tokens
    end
  end
  self:process(tlines,tokens,suponly) -- the missing training instance
  return self.desc, self.qs, self.ans
end

function babi:process(tlines,qs,suponly)--tlines is map. qs is list
  if #qs ==0 then return end
  local desc = tds.vec()
  local question = tds.vec()
  local answer = tds.vec()

  --  print(tlines) print(qs)
  local answerparts = stringx.split(stringx.strip(qs[#qs]),'\t')
  --  print(answerparts)
  local ans= answerparts[1]

  if suponly then
    local supids = stringx.split(answerparts[2],',')
    --  print(supids)
    local start = 1
    for _, supid in pairs(supids) do
      if start ==1 then
        start = 0
      else
        desc:insert(0)
      end
      local dline = tlines[tonumber(supid)]
      for i=2, #dline do
        desc:insert(self.desc_vocab[dline[i]])
      end
    end
  else
    local start = 1
    for _, words in pairs(tlines) do
      if start ==1 then
        start = 0
      else
        desc:insert(0)
      end
      for i=2, #words do
        desc:insert( self.desc_vocab[words[i]] )
      end
    end
  end

  for i=2, #qs-1 do
    --    print(qs[i])
    question:insert(self.desc_vocab[qs[i]])
  end

  answer:insert(self.q_vocab[ans])
  --
  --  print(desc)
  --  print(question)
  --  print(answer)

  table.insert(self.desc,desc)
  table.insert(self.qs, question)
  table.insert(self.ans, answer)
end


