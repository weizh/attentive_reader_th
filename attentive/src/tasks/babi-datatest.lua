
require 'babi' 

bb = BABI()

local ds, qs, as = bb:readFile('data/qa1_train.txt', true,false) --- is training?  supporting facts only?
local dst, qst, ast = bb:readFile('data/qa1_test.txt',false,false)

print(#ds .. ' ' .. #qs .. ' ' .. #as)
print(#dst .. ' ' .. #qst .. ' ' .. #ast)

print (ds)
