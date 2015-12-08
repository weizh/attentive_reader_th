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
  end
  --  print(fname)
  local tlines = {}
  local parts
  local tokens
  for line in io.lines(fname) do  --print(line)
    parts = stringx.split(stringx.strip(line), '\t')
    dtokens = stringx.split(stringx.strip(parts[1]))
    if tonumber(dtokens[1]) ==1 then -- a new instance
      tlines[tonumber(dtokens[1])]= dtokens  -- reset tlines
    elseif #parts>1 then
      self:process(tlines,dtokens,parts[2],parts[3],suponly) --parts[2] answer, 3 is fids.
    else
      tlines[tonumber(dtokens[1])] = dtokens
    end
  end
  self:process(tlines,dtokens,parts[2],parts[3],suponly) -- the missing training instance
  return self.desc, self.qs, self.ans
end

function babi:process(tdlines,tqline,ans,fid,suponly)--tlines is map. qs is list
  local desc = tds.vec()
  local question = tds.vec()
  local answer = tds.vec()

  local fids = stringx.split(stringx.strip(fid),' ')

  if suponly then
    local start = 1
    for _, supid in pairs(fids) do
      if start ==1 then
        start = 0
      else
        desc:insert(0)
      end
      local dline = tdlines[tonumber(supid)]
      for i=2, #dline do
        desc:insert(self.desc_vocab[dline[i]])
      end
    end
  else
    local start = 1
    for i, words in pairs(tdlines) do
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

  for i=2, #tqline do
    --    print(qs[i])
    question:insert(self.desc_vocab[tqline[i]])
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


