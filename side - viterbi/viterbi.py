from hmm import *


class my_viterbi():
    def __init__(self, observation, states, trans_p, emit_p, data='WSJ_24.pos'):
        self.words = observation
        self.tags = states
        self.transition = trans_p
        self.emission = emit_p
        self.init_prob = None

        self.data_path = 'WSJ_POS_CORPUS_FOR_STUDENTS/{}'.format(data)
        self.data = []
        self.num_data = 0

        self.num_words = len(self.words)
        self.num_tags = len(self.tags)
        self.tag2idx, self.word2idx, self.idx2tag, self.idx2word = None, None, None, None
        self.tot_path = []

    def init(self):
        # load data
        self.data_loader()

        # get dictionary
        words_idx = np.arange(self.num_words)
        tags_idx = np.arange(self.num_tags) 
        self.tag2idx = dict(zip(self.tags, tags_idx))
        self.word2idx = dict(zip(self.words, words_idx))
        self.idx2tag = dict(zip(tags_idx, self.tags))
        self.idx2word = dict(zip(words_idx, self.words))

        # get initial probability
        init_idx = self.tag2idx['<init>']
        self.init_prob = self.transition[init_idx].reshape([-1, 1])

    def data_loader(self):
        data = open(self.data_path, 'r')
        sentence = []
        for t in data:
            token = t.strip()
            if not token == "":
                sentence.append(token)
            else:
                self.data.append(np.array(sentence))
                sentence = []

        self.data = np.array(self.data)
        self.num_data = len(self.data)

    def calculation(self, sentence):
        T = len(sentence)  # length of sentence
        prob = np.empty(shape=[self.num_tags, len(sentence)])
        path = np.zeros(shape=[self.num_tags, len(sentence)])

        # initial assignment t=0
        for i in range(self.num_tags):
            if self.init_prob[i] == 0:
                prob[i][0] = float("-inf")
                path[i][0] = 0
            else:
                if sentence[0] in self.words:
                    realization = sentence[0]
                else:
                    realization = '<unk>'
                prob[i][0] = np.log(self.init_prob[i]) + np.log(self.emission[i][self.word2idx[realization]])
                path[i][0] = 0

        # t = 1,2, ..., T
        for t in range(1, T):

            # current tag
            for i in range(self.num_tags):
                best_prob = float("-inf")
                best_path = None

                # previous tag
                for j in range(self.num_tags):
                    if sentence[t] in self.words:
                        realization = sentence[t]
                    else:
                        realization = '<unk>'
                    crnt_prob = prob[j][t - 1] + np.log(self.transition[j][i]) \
                                + np.log(self.emission[i][self.word2idx[realization]])

                    if crnt_prob > best_prob:
                        best_prob = crnt_prob
                        best_path = j

                prob[i][t] = best_prob
                path[i][t] = best_path

        return prob, path, T

    def trace_back(self, prob, path, time_step):
        trace = []
        final_state = float('-inf')

        # t = T
        for i in range(0, self.num_tags):
            if prob[i][time_step - 1] > final_state:
                final_state = prob[i][time_step - 1]
                trace.append(i)

        # t = T-1, T-2, ..., 2, 1, 0
        for j in range(time_step - 1, 0, -1):
            trace.append(path[int(trace[-1])][j])

        self.tot_path.append(trace[::-1])
        pass

    def running(self):
        self.init()
        ns = []
        for n, sent in enumerate(self.data):
            if n % 1 == 0:
                print('{0}th: {1}\n'.format(n, sent))
                prob, path, time_step = self.calculation(sent)
                self.trace_back(prob, path, time_step)
                ns.append(n)

        pred_pos = []
        for i in range(len(self.tot_path)):
            pred_pos.append([self.idx2tag[idx] for idx in self.tot_path[i]])

        return pred_pos, ns

    def save_results(self, ns, pos, is_test):
        if not os.path.exists('results'):
            os.mkdir('results')

        if is_test:
            path = 'results/WSJ_23.pos'
        else:
            path = 'results/prediction.txt'

        output = open(path, 'w')
        for j, i in enumerate(ns):
            for token, tag in zip(self.data[i], pos[j]):
                line = '{}\t{}\n'.format(token, tag)
                output.write(line)
            output.write("\n")
        output.close()

    def save_true(self, ns, true_file='WSJ_24.pos'):
        in_file = open('WSJ_POS_CORPUS_FOR_STUDENTS/{}'.format(true_file), 'r')
        sentence = []
        whole_data = []
        for t in in_file:
            token = t.strip()
            if not token == "":
                sentence.append(token)
            else:
                whole_data.append(np.array(sentence))
                sentence = []

        out_file = open('results/true.txt', 'w')
        for i in ns:
            for token_tag in whole_data[i]:
                line = '{}\n'.format(token_tag)
                out_file.write(line)
            out_file.write("\n")
        out_file.close()


if __name__ == "__main__":
    hmm = my_hmm(min_cnt=2, train_data='wsj_02-21.pos', smoothing=0.001)
    observation, states, trans_p, emit_p = hmm.train()
    viterbi = my_viterbi(observation, states, trans_p, emit_p, 'WSJ_24.words')
    prediction, ns = viterbi.running()
    # viterbi.save_results(ns, prediction)
    viterbi.save_true(ns, true_file='WSJ_24.pos')
