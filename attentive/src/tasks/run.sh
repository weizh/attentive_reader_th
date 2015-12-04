
# the parameter setting is to follow deepmind's paper

Substitue the train and valid data set will be just fine. 
th rc-task.lua --train data/sample.txt --valid data/sample.txt --cont_dim 256 --m_dim 256 --g_dim 256 --dropout 0.2 --full_vocab_output --eval_interval 1000 
