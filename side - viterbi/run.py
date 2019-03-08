from viterbi import *
from scorer import *

hmm = my_hmm(min_cnt=2, train_data='wsj_02-21.pos', smoothing=0.001)
observation, states, trans_p, emit_p = hmm.train()

dev_viterbi = my_viterbi(observation, states, trans_p, emit_p, 'WSJ_24.words')
prediction, ns = dev_viterbi.running()
dev_viterbi.save_results(ns, prediction, is_test=False)
dev_viterbi.save_true(ns, true_file='WSJ_24.pos')
score('results/true.txt', 'results/prediction.txt')

test_viterbi = my_viterbi(observation, states, trans_p, emit_p, 'WSJ_23.words')
prediction, ns = test_viterbi.running()
test_viterbi.save_results(ns, prediction, is_test=True)
