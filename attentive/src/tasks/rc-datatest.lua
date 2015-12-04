require 'Data'

local data = Data()

local desc, q, a = data:read('data/temp.txt',true,false)
print('train done')
local tdesc, tq, ta = data:read('data/temp.txt',false,false)
print('valid done')
print (#desc)
print (#q)
print(#a)

print (#tdesc)
print (#tq)
print(#ta)

function toStringm(m)
  for i =1, m:size(1) do
    for j = 1, m:size(2) do
      print(data:getiVocab()[m[i][j]])
    end
  end
end
function toStringV(m)
  for i=1,m:size(1) do
    print(data:getiVocab()[m[i]])
  end
end

function toStringS(m)
  print(data:getEiVocab()[m])
end

--print(#data:getVocab())
--print(data:getVocab())

for i = #tq, #tq do
  print (string.rep('--- desc ',30))
  print (tdesc[i])
  toStringm(tdesc[i])
  print (string.rep('--- q ',20))
  print (tq[i])
  toStringV(tq[i])
  print (string.rep('--- a ',20))
  print (ta[i])
  toStringS(ta[i])
end

print(string.rep('-', 100))
print(#data:getEvec())
print(data:getEvec())

print(#data:getEVocab())
print(data:getEVocab())

print(#data:getEiVocab())
print(data:getEiVocab())

--print(a)


local ind = 1
local i = 0
--for ind=1, #desc do
while true do
  if i==0 then break end
  print ('----------------------------------------------the ' .. i ..' th permutation------------------------------------------------------------')
  print('----------- the chosen item is --------------')
    print('--------desc')
  --  print( desc[ind])
      toStringm(desc[ind])
    print('-------question')
--    print(q[ind])
      toStringV(q[ind])
    print('--------answer')
  --  print( a[ind])
      toStringS(a[ind])

  local found = false
--  print(#desc[ind])
  local m = desc[ind]
  for i =1, m:size(1) do
    for j = 1, m:size(2) do
      if data:getEiVocab()[a[ind]]== data:getiVocab()[m[i][j]] then
        found = true
      end
    end
  end
  if found == false then
    error('Error occured')
  end
  print('----------- permuting ... ------------')

  local nd,nq,na =data:permute_an_example(desc,q,a,ind)
  print('----------- the permuted item is ------------')
  print('--------desc')
--  print( nd)
  toStringm(nd)
  print('-------question')
--  print(nq)
   toStringV(nq)
  print('--------answer')
--  print( na)
  toStringS(na)
--  print(string.rep('-',30) .. 'shuffled entities are : ')
--  print(data:getMappedEntites())
  i = i+1
  if i > 1000 then break end
end
