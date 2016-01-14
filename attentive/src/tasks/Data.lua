-- zhangwei
-- The model that reads in desc, question and answer, and store them into
-- three separate tables.
-- The length of desc table is nlines times bigger than question or answer.

--Note that entity list is additional, and only for result dict, not for
--  context dict.

local Data = torch.class('Data')

local stringx = require('pl.stringx')
local file = require('pl.file')
local tds = require('tds')

function Data:__init()
  self.vocab =  tds.hash()
  self.ivocab =  tds.hash()
  self.freq_vocab = tds.hash()
  self.e_vocab = tds.hash()
  self.e_ivocab = tds.hash()
  self.e_vec = tds.vec()
  self.vocab['<unk>']=1
  self.ivocab[1] ='<unk>'
  self.e_vocab['<unk>']=1
  self.e_ivocab[1]='<unk>'
  self.unkidx = 1
  self.is_full_vocab = true
end

--- entites are both in vocab and e_vocab.
-- e_ivocab contains inverted index of e_vocab.
-- ivocab contains inverted index of entities, in vocab.
-- depend on where you use entity (in input or ouput), the index will be
-- different.

function Data:read(fname,train, is_full_vocab)
  self.is_full_vocab = is_full_vocab
  local desc ={}
  local question = {}
  local answer = {}
  local temp_desc_str = {}
  --  print ('creating dictionary from training data. Unigrams are mapped to  unknown.')
  if train then
    for line in io.lines(fname) do
      local w = stringx.split(line)
      for i = 2,#w do
        if not self.freq_vocab[w[i]] then self.freq_vocab[w[i]]=0 end   ---freq_vocab is used to store the frequencies of entities. Unigrams are mapped to unknown.
        self.freq_vocab[w[i]]=self.freq_vocab[w[i]]+1
      end
    end
  end

  -- print('reading lines')

  for line in io.lines(fname) do
    local w = stringx.split(line)
    local num = tonumber(w[1])

    if train then
      for i = 2,#w do
        if string.find(w[i],'@entity') then       ----write e_vocab in the first pass. e_vocab and e_ivocab is indexed by its own index.
          if not self.e_vocab[w[i]] then
            self.e_vocab[w[i]]= #self.e_vocab + 1
            self.e_ivocab[#self.e_vocab]=w[i]
          end
        end
        if self.e_vocab[w[i]] or self.freq_vocab[w[i]] >1 then   --- Second pass, if word is an entity, add it to dict if not in dict.
          if not self.vocab[w[i]] then
            self.vocab[w[i]] = #self.vocab+1
            self.ivocab[#self.vocab] = w[i]
          end
        --- note: the rest freq ==1 words are ignored, because <unk> is  already in dic.
        end
      end
    end

    if num==1 then-- if it is the first line, which is in turn not a question line
      if #temp_desc_str ~= 0 then -- handle previously collected description instances.
        self:process(temp_desc_str,desc,question,answer,train)
    end
    temp_desc_str = {}
    end
    table.insert(temp_desc_str,w) -- add the line anyways
  end
  -- handle the last missing example
  self:process (temp_desc_str,desc,question,answer,train)

  if train then
    -- get the entity vector as auxilary data structure for future use
    self:get_entity_keys()
  end

  return desc,question,answer
end

function Data:get_entity_keys()
  for k, v in pairs(self.e_vocab) do
    self.e_vec:insert(k)
  end
end

function Data:process(temp_desc_str,desc,question,answer,train)
  --print(temp_desc_str)
  local maxSentLen = 0
  local document_size = #temp_desc_str-1
  for i=1,document_size do
    if maxSentLen < #temp_desc_str[i]-1 then maxSentLen = #temp_desc_str[i]-1
    end  --- exclude question mark
  end

  local indmatrix = torch.zeros(document_size, maxSentLen)

  ----------------- if it is validation set -------------------------------

  local tempMap = tds.hash() -- to store maps ORIGINAL-MAPPED entities in validation or test set. NOTE: All entitites in ORIGINAL, no matter if it is OOEV or not, are mapped.

  if not train then
    
    local entities = tds.hash() -- to store all the entities

    for i=1,#temp_desc_str do   --- loop over all sents, including question.
      local toks = temp_desc_str[i]
      for j = 2, #toks do
        if string.find(toks[j],'@entity') then entities[toks[j]]=1 end   ---collect all the entities
      end
    end

    local mapped_eids = tds.hash()  -- This is to store the entities that are  mapped to. And mapped-to entities should not be used again for other mapping entities. NOTE: IDs are vocab ids, NOT e_vocab ids.

    if #entities~=0 then
      for k,v in pairs(entities) do
        local estr = self.e_vec[math.floor(torch.uniform(1,#self.e_vec+0.99))]
        local d = self.vocab[ estr ]
        while  mapped_eids[d] do
          estr = self.e_vec[math.floor(torch.uniform(1,#self.e_vec+0.99))]
          d = self.vocab[ estr ]
        end
        tempMap[ k ] = d
        mapped_eids [d ] = 1
      end
    end
  end
  ----------------- description ---------------------------------
  for i=1,document_size do
    local toks = temp_desc_str[i]
    for j = 2, #toks do
      local w = toks[j]
      if train then -- if it's training, the words are guaranteed in dictionary.
        if self.vocab[w] then
          indmatrix[i][j-1] = self.vocab[w]
        else                   -- if it's not in the vocabulary, meaning it's a  rare word.
          indmatrix[i][j-1] = self.unkidx
        end
      else   --- if it is validation data or test data
        if string.find(toks[j],'@entity') then    --- if it is an entity
	  indmatrix[i][j-1] = tempMap[w]
        elseif self.vocab[w] then
          indmatrix[i][j-1] = self.vocab[w]
        else
          indmatrix[i][j-1] = self.unkidx
      end
      end
    end
  end

  ------------------  question --------------------------
  local ql = temp_desc_str[#temp_desc_str]
  local qt = torch.zeros( #ql - 4 )

  for i=2, #ql-3 do
    local w= ql[i]
    if train then  -- in training, everything is in vocabulary. if not, it is unk.
      if self.vocab[w] then
        qt[i-1] = self.vocab[w]
      else
        qt[i-1] = self.unkidx
      end
    else  -- in test, words are not guaranteed to be in vocab.
      if string.find(ql[i],'@entity') then   --- if entity not in vocab, generate a random entity in vocab.
        qt[i-1] = tempMap[w]
      elseif self.vocab[w] then
        qt[i-1] = self.vocab[w]
      else
        qt[i-1] = self.unkidx
      end
    end
  end

  -----  answer -------
  local a4r
  local aw = ql[#ql-1]
  if train then --- if it's in train mode
	if not self.vocab[aw] then
		return
 	end
	if self.is_full_vocab then
	    a4r = self.vocab[aw]
        else
            a4r = self.e_vocab[aw]
        end
  else  --- if it's in validation or test mode
	if self.is_full_vocab then
		a4r = tempMap[aw] --- The entity should have been mapped to another entity id already.
	else
 	        a4r= self.e_vocab[ self.ivocab[tempMap[aw]] ]
   	end
  end
  
  if a4r == nil or aw =='null' then
    print("Found a Nil answer! Skipped!")
    print(temp_desc_str)
  else
    table.insert(desc, indmatrix)
    table.insert(question, qt)
    table.insert(answer,a4r)
  end
end

-- only permute on training, should never be done on test.
function Data:permute_an_example(descs, questions, answers, index)
  local d = descs[index]
  local q = questions[index]
  local a = answers[index]

  local enids = tds.hash()
  ------------------------------- get entities first
  --------------------------------------------
  local m = d:size()
  for i = 1, m[1] do
    for j=1, m[2] do
      if d[i][j] ~=0 then
        if self.e_vocab[ self.ivocab[ d[i][j] ] ] then -- if the word is a entity, swap it.
          enids[d[i][j]] =1
        end
      end
    end
  end

  for j=1, q:size(1) do
    if q[j] ~=0 then
      if self.e_vocab[ self.ivocab[ q[j] ] ] then
        enids[ q[j] ] =1
      end
    end
  end

  --------------------------------- permute
                                    ---------------------------------------------------
  --permute entity list, using map

  local amap = tds.hash()
  local shuf_eids = tds.hash()
  for k,_ in pairs(enids) do
    local d = self.vocab[
  self.e_vec[math.floor(torch.uniform(1,#self.e_vec+0.99))] ]
    while shuf_eids[d] do
      d = self.vocab[
  self.e_vec[math.floor(torch.uniform(1,#self.e_vec+0.99))] ]
    end
    amap[ k ] = d
    shuf_eids [d ] = 1
  end

  ----------------------------------write back
                                    ------------------------------------------------
  local m = d:size()
  for i = 1, m[1] do
    for j=1, m[2] do
      if d[i][j] ~=0 then
        if self.e_vocab[ self.ivocab[ d[i][j] ] ] then -- if the word is entity, swap it.
          d[i][j] = amap[ d[i][j] ];
        end
      end
    end
  end

  for j=1, q:size(1) do
    if self.e_vocab[ self.ivocab[ q[j] ] ] then
      q[j] = amap[ q[j] ];
    end
  end

  --- the answer is guaranteed since the permutation is on training example. So find permuted output entity id by reverse lookup.

  if not self.is_full_vocab then
    local ind = self.vocab[ self.e_ivocab[a] ]
    local mapped_ind = amap[ind]
    local mapped_str = self.ivocab[ mapped_ind ]
    a = self.e_vocab[mapped_str];
    answers[index] = a -- move answers since it's a table, a is not a reference.
    return d,q,a
  else
    answers[index] = amap[a]
    return d,q,amap[a]
  end
end

function Data:getVocab()
  return self.vocab
end

function Data:getiVocab()
  return self.ivocab
end

function Data:getEVocab()
  return self.e_vocab
end
function Data:getEiVocab()
  return self.e_ivocab
end

function Data:getEvec()
  return self.e_vec
end
