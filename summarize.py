# coding=UTF-8
from __future__ import division
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import brown, wordnet as wn, stopwords
import nltk, math, re, sys, numpy as np

ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

def get_best_synset_pair(word_1, word_2):
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = synset_1.wup_similarity(synset_2)         # wu palmer metric for computing similarity
               #sim = wn.path_similarity(synset_1, synset_2)
               if sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair

def length_dist(synset_1, synset_2):
    l_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            l_dist = 1.0
        else:
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    return math.exp(-ALPHA * l_dist)

def hierarchy_dist(synset_1, synset_2):
    h_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if hypernyms_1.has_key(lcs_candidate):
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if hypernyms_2.has_key(lcs_candidate):
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * hierarchy_dist(synset_pair[0], synset_pair[1]))

def most_similar_word(word, word_set):
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim
    
def info_content(lookup_word):
    global N
    if N == 0:
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if not brown_freqs.has_key(word):
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if not brown_freqs.has_key(lookup_word) else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))
    
def semantic_vector(words, joint_words, info_content_norm):
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec                
            
def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))

    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)

    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

def word_order_vector(words, joint_words, windex):
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            wovec[i] = windex[joint_word]
        else:
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def word_order_similarity(sentence_1, sentence_2):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))


def similarity(sentence_1, sentence_2, info_content_norm):
    return (DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2))

class SummaryTool(object):
    def split_content_to_paragraphs(self, content):
        return content.split("\n\n")

    def format_sentence(self, sentence):
        sentence = re.sub(r'\W+', '', sentence)
        return sentence

    def get_sentences_ranks(self, content):

        sentences = nltk.sent_tokenize(content.decode('utf-8'))
        n = len(sentences)
        values = [[0 for x in xrange(n)] for x in xrange(n)]
        
        for i in range(0, n):
            for j in range(0, i):
                if i == j:
                    values[i][j] = 1.0
                else:
                    values[i][j] = values[j][i] = similarity(sentences[i], sentences[j], False)

        sentences_dic = {}
        for i in range(0, n):
            score = 0
            for j in range(0, n):
                if i == j:
                    continue
                score += values[i][j]
            sentences_dic[self.format_sentence(sentences[i])] = score
        
        return sentences_dic

    def cue_score(self, content, stop_words):
        sentences = nltk.sent_tokenize(content)
        cue_scores = []

        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            count = [w for w in words if w in stop_words]
            score = 1 - (float(len(count))/len(words))
            cue_scores.append(score)

        return cue_scores

    def location_score(self, content):
        paragraphs = self.split_content_to_paragraphs(content)
        scores = []
        for para in paragraphs:
            sent = nltk.sent_tokenize(para)
            L = len(sent)
            mean = L/2.0
            for i in range(0, L):
		scores.append(float(abs(i - mean))/L + 0.01)

        return scores

    def length_score(self, content):
        Min = 3
        Max = 7
        mid = (Max + Min)/2.0
        variance = 2.0
        sentences = nltk.sent_tokenize(content)
        scores = []
        for each in sentences:
            cut_off = len(nltk.word_tokenize(each))
            temp = -((cut_off - mid)**2)/2.0
            scores.append(math.exp(temp/variance))

        return scores

    def get_summary(self, content, ranking_scores, loc_score, len_score, cue_words_score):

        sentences = nltk.sent_tokenize(content)
        summary = []
        final_sum = []

        # weights for individual features : ranking --> 0.336 , location --> 0.144 , length --> 0.32 , cue_words --> 0.2
        p = 0.7
        q = 0.6
        r = 0.8

        for i in range(0, len(sentences)):
            strip_s = self.format_sentence(sentences[i])
            Sum = 0

            if strip_s:
                Sum += (p * q * r * ranking_scores[strip_s])
            Sum += ( (r*q*(1-p)*loc_score[i]) + (r*(1-q)*len_score[i]) + ((1-r)*cue_words_score[i]) )
            
            final_sum.append(Sum)

        keys = sorted(range(len(final_sum)), key = lambda i: final_sum[i])[-15 : ]
        #print keys
        for each in keys :
            summary.append(sentences[each])

        return ("\n").join(summary)

    def sentiment(self, content):
        sid = SentimentIntensityAnalyzer()
        sentences = nltk.sent_tokenize(content)
        for each in sentences:
            print each,
            print sid.polarity_scores(each)


def main():
    stop_words = set(stopwords.words('english'))

    text_file = open("Document.txt", "r")
    lines = text_file.readlines()

    content = ""
    for each in lines:
        content = content + each
    
    st = SummaryTool()

    ranking_scores = st.get_sentences_ranks(content)
    cue_words_score = st.cue_score(content, stop_words)
    loc_score = st.location_score(content)
    len_score = st.length_score(content)

    summary = st.get_summary(content, ranking_scores, loc_score, len_score, cue_words_score)
    #print summary

    fact = "Blood is an essential component"
    fact1 = "Blood banks maintain stock of blood"
    Fact_checking = []
    Fact_checker = []

    for each in summary.split("\n"):
        Fact_checking.append(similarity(fact, each, False))
        Fact_checker.append(similarity(fact1, each, False))

    print sum(Fact_checking), sum(Fact_checker)

    st.sentiment(summary)

    st.sentiment(fact)
    st.sentiment(fact1)


if __name__ == '__main__':
    main()
