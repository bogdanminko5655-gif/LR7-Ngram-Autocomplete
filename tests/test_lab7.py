"""Тести для ЛР №7. Запуск: python -m unittest discover -s tests -v"""
import math
import unittest
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab7 import (START, END, UNK, get_tokenized_data, count_words,
                  get_words_with_nplus_frequency, replace_oov_words_by_unk,
                  count_n_grams, NgramLanguageModel, estimate_probability)

DATA = [['i', 'love', 'cats'], ['i', 'love', 'dogs'], ['i', 'see', 'cats'], ['you','love','cats']]
VOCAB = sorted(set(w for sent in DATA for w in sent))

class Tests(unittest.TestCase):
    def setUp(self):
        self.m = NgramLanguageModel.fit(DATA, VOCAB, k=0.5)

    def test_tokenization(self):
        self.assertEqual(get_tokenized_data("HELLO, there!\nDon't go."), [['hello','there'], ["don't", 'go']])

    def test_freq(self):
        self.assertEqual(count_words(DATA)['cats'], 3)
        self.assertNotIn('dogs', get_words_with_nplus_frequency(DATA, 2))

    def test_unk(self):
        self.assertEqual(replace_oov_words_by_unk([['hello', 'earth']], ['hello']), [['hello', UNK]])

    def test_grams(self):
        grams = count_n_grams([['i','love']], 3)
        self.assertEqual(grams[(START, START, 'i')], 1)
        self.assertEqual(grams[('i', 'love', END)], 1)

    def test_transition_normalization(self):
        for n, context in [(1,[]),(2,['i']),(3,['i','love']),(5,['foo'])]:
            total = sum(self.m.probability(w,context,n) for w in self.m.vocabulary)
            self.assertAlmostEqual(total,1,places=10)

    def test_bigram_prefers_observed(self):
        self.assertGreater(self.m.probability('love',['i'],2), self.m.probability('dogs',['i'],2))

    def test_trigram_prefers_cats(self):
        self.assertGreater(self.m.probability('cats',['i','love'],3), self.m.probability('see',['i','love'],3))

    def test_undefined_word_maps_to_unk(self):
        self.assertEqual(self.m.probability('foo',['i'],2), self.m.probability(UNK,['i'],2))

    def test_suggestions_prefix(self):
        top, actual = self.m.suggest('i love',n=3,top_k=3,start_with='c')
        self.assertEqual(top[0][0], 'cats')
        self.assertTrue(all(w.startswith('c') for w,_ in top))
        self.assertEqual(actual, 3)

    def test_backoff(self):
        _, actual = self.m.suggest('unspecified zzz',n=5)
        self.assertEqual(actual,1)

    def test_perplexity_positive(self):
        for n in range(1,6):
            self.assertGreaterEqual(self.m.perplexity(DATA,n),1.0)
            self.assertTrue(math.isfinite(self.m.perplexity(DATA,n)))

    def test_probability_reference(self):
        ngrams = Counter({('i', 'love'):2,('i','see'):1})
        prob = estimate_probability('love', ('i',), {}, ngrams, 5, k=1)
        self.assertAlmostEqual(prob, 3/8)

if __name__ == '__main__':
    unittest.main()
