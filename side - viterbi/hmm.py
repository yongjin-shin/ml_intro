from collections import defaultdict
import os
import string
import numpy as np


class my_hmm():
    def __init__(self, min_cnt=2, train_data='wsj_02-21.pos', smoothing=0.001):
        self.path = 'WSJ_POS_CORPUS_FOR_STUDENTS/{}'.format(train_data)
        self.train_data = open(self.path, "r")
        self.min_cnt = min_cnt
        self.num_tags = 0
        self.num_words = 0
        self._lambda = smoothing

        # save data
        self.vocab = None
        self.words = []
        self.tag = None
        self.trans = defaultdict(int)  # transition
        self.emisn = defaultdict(int)  # emission
        self.cntag = defaultdict(int)  # count tag
        self.transition = None
        self.emission = None

        # about unknown - check def assign_unk
        self.punct = set(string.punctuation)
        self.noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist",
                            "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
        self.verb_suffix = ["ate", "ify", "ise", "ize"]
        self.adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
        self.adv_suffix = ["ward", "wards", "wise"]

    def get_vocab_tag(self):
        v, t = defaultdict(int), []  # vocabulary, tags
        for line in self.train_data:
            if not line.split():
                continue  # new line

            token, tag = line.strip().split("\t")
            v[token] += 1
            t.append(tag)

        v = [token for token, cnt in v.items() if cnt >= self.min_cnt]
        self.vocab = sorted(v)
        self.tag = np.unique(t)
        self.num_words = len(self.vocab)
        print("Finished - Get vocabulary and tags")

    def count_tec(self):
        # v = set(self.vocab)  # vocab
        self.train_data = open(self.path, "r")
        prev = "<init>"

        unk_output = open("data/unk_output.txt", "w")
        what_is_wrong_with = []

        for line in self.train_data:
            if not line.split():
                tag = "<init>"
                word = "<empty>"
            else:
                word, tag = line.split()
                if word not in self.vocab:
                    # what_is_wrong_with.append("{0} {1}".format(word, tag))
                    word = '<unk>'
                    # word = self.assign_unk(word)

            self.trans[" ".join([prev, tag])] += 1
            self.emisn[" ".join([tag, word])] += 1
            self.cntag[tag] += 1
            self.words.append(word)
            prev = tag

        for tmp in what_is_wrong_with:
            unk_output.write(tmp+'\n')
        unk_output.close()
        print("Finished - count transition/emission/count tags")

    def save_model(self):
        if not os.path.exists('data'):
            os.mkdir('data')

        output = open("data/vocab.txt", "w")
        for word in self.vocab:
            output.write("{0}\n".format(word))
        output.close()

        output = open("data/tag.txt", "w")
        for tag in self.tag:
            output.write("{0}\n".format(tag))
        output.close()

        output = open("data/model.txt", "w")
        for k, v in self.trans.items():
            line = "Trans {0} {1}\n".format(k, v)
            # self.model.append(line)
            output.write(line)

        for k, v in self.emisn.items():
            line = "Emiss {0} {1}\n".format(k, v)
            # self.model.append(line)
            output.write(line)

        for k, v in self.cntag.items():
            line = "Cntag {0} {1}\n".format(k, v)
            # self.model.append(line)
            output.write(line)
        output.close()

    def load_model(self, path):
        self.trans = defaultdict(dict)  # transition
        self.emisn = defaultdict(dict)  # emission
        self.cntag = defaultdict(dict)  # count tag

        model = open(path, 'r')
        self.words = []
        for line in model:
            if line.startswith("C"):
                _type, tag, count = line.split()
                self.cntag[tag] = int(count)
                continue

            _type, tag, tag_word, count = line.split()
            if _type == "Trans":
                self.trans[tag][tag_word] = int(count)
            else:
                self.emisn[tag][tag_word] = int(count)
                self.words.append(tag_word)

        print("Finished - Load Model")

    def do_some_process(self):
        old_tag = set(np.copy(self.tag))
        self.tag = list(self.cntag.keys())
        print("Diff tag is {}".format(set(self.tag) - old_tag))
        self.num_tags = len(self.tag)  # some special tags were added

        t = open('data/unk_output.txt', 'r')
        b = []
        for i in t:
            t, p = i.strip().split()
            b.append(t)
        print("Diff Words is {}".format(set(self.words) - set(self.vocab) - set(b)))
        self.num_words = len(self.words)  # some special words were added

    def get_transition_prob(self):
        self.transition = np.zeros(shape=[self.num_tags, self.num_tags], dtype=np.float32)

        for i, prev_tag in enumerate(self.tag):
            for j, curnt_tag in enumerate(self.tag):
                value = 0
                # if prev_tag == 'PRP' and curnt_tag == 'WP$':
                #     print("")
                if curnt_tag in self.trans[prev_tag]:
                    value = self.trans[prev_tag][curnt_tag]
                self.transition[i][j] = value + self._lambda  # /self.cntag[prev_tag]

        row_sum = np.sum(self.transition, axis=1)
        # self.transition = self.transition/np.array(list(self.cntag.values())).reshape([-1, 1])
        self.transition = self.transition/np.reshape(row_sum, newshape=(-1, 1))

        print("Finished - Transition Prob")

    def get_emission_prob(self):
        self.emission = np.zeros(shape=[self.num_tags, self.num_words], dtype=np.float32)

        for i, curnt_tag in enumerate(self.tag):
            for j, curnt_vocab in enumerate(self.words):
                value = 0
                if curnt_vocab in self.emisn[curnt_tag]:
                    value = self.emisn[curnt_tag][curnt_vocab]
                self.emission[i][j] = value + self._lambda  # /self.cntag[curnt_tag]

        row_sum = np.sum(self.emission, axis=1)
        # self.emission = self.emission / np.array(list(self.cntag.values())).reshape([-1, 1])
        self.emission = self.emission / np.reshape(row_sum, newshape=(-1, 1))

        print("Finished - Emission Prob")

    # about unknown
    def assign_unk(self, tok):
        if any(char.isdigit() for char in tok):  # Digits
            return "--unk_digit--"
        elif any(char in self.punct for char in tok):  # Punctuation
            return "--unk_punct--"
        elif any(char.isupper() for char in tok):  # Upper-case
            return "--unk_upper--"
        elif any(tok.endswith(suffix) for suffix in self.noun_suffix):  # Nouns
            return "--unk_noun--"
        elif any(tok.endswith(suffix) for suffix in self.verb_suffix):  # Verbs
            return "--unk_verb--"
        elif any(tok.endswith(suffix) for suffix in self.adj_suffix):  # Adjectives
            return "--unk_adj--"
        elif any(tok.endswith(suffix) for suffix in self.adv_suffix):  # Adverbs
            return "--unk_adv--"
        return "--unk--"

    def train(self):
        self.get_vocab_tag()

        if os.path.exists('data/model.txt'):
            self.load_model('data/model.txt')
        else:
            self.count_tec()
            self.save_model()
            self.load_model('data/model.txt')

        self.do_some_process()
        self.get_transition_prob()
        self.get_emission_prob()
        return self.words, self.tag, self.transition, self.emission


if __name__ == "__main__":
    hmm = my_hmm(min_cnt=2, train_data='wsj_02-21.pos', smoothing=0.001)
    observation, states, trans_p, emit_p = hmm.train()
