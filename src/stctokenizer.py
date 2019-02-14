import keras
import unicodedata
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import nltk
import re
# nltk.download('punkt')


class STCTokenizer:
    def __init__(self):
        self.half_filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~   '
        self.full_filters = '！＂＃＄％＆（）＊＋，－．／：；＜＝＞？＠［＼］︿＿‵｛｜｝～　“”’'
        self.stopwords = stopwords.words("english")
        self.postfixdict = {'m': 'am',
                            's': 'is',
                            're': 'are',
                            've': 'have',
                            'll': 'will', }

    def tokenize(self, type, text, remove_stopwords, to_lower):
        if not isinstance(text, str):
            raise TypeError('1st argument: text, must be str')
        if not isinstance(remove_stopwords, bool):
            raise TypeError('2nd argument: remove_stopwords, must be bool')

        text = unicodedata.normalize('NFKC', text)
        # text = re.sub(r'[^A-Za-z\']', '', text)

        if remove_stopwords:
            text = re.sub(r'[^A-Za-z ]', '', text)
        else:
            text = re.sub(r'[^A-Za-z!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ \']', '', text)

        if type == 'nltk':
            filter_tokens = nltk.word_tokenize(text)

        elif type == 'keras':

            tokens = keras.preprocessing.text.text_to_word_sequence(
                text,
                # filters=self.half_filters + self.full_filters,
                lower=to_lower,
                split=' '
            )

            filter_tokens = []
            for idx, word in enumerate(tokens):
                if "'" in word and not word.endswith("'"):
                    word_split = word.split("'")
                    length = len(word_split)
                    if length != 2:
                        filter_tokens += [word.replace("'", "")]
                        continue
                    # assert length == 2, 'Error word format {} -> {}, length = {}'.format(word, word_split, length)
                    prefix, postfix = word_split
                    if postfix == 't':
                        if prefix == 'shan':
                            filter_tokens += ['shall', 'not']
                        elif prefix == 'can':
                            filter_tokens += ['can', 'not']
                        elif prefix == 'won':
                            filter_tokens += ['will', 'not']
                        else:
                            filter_tokens += [prefix, postfix]
                    elif postfix == 'd':
                        if idx >= len(tokens) - 2:
                            filter_tokens += [prefix]
                            continue
                        elif tokens[idx + 1] == 'better' or tokens[idx + 1] == 'best':
                            filter_tokens += [prefix, 'had']
                        else:
                            filter_tokens += [prefix, 'would']
                    elif word == "let's":
                        filter_tokens += ['let', 'us']
                    elif postfix in self.postfixdict.keys():
                        filter_tokens += [prefix, self.postfixdict[postfix]]
                    else:
                        pass
                else:
                    filter_tokens += [word]

        else:
            raise NameError('Wrong type: {}'.format(type))

        if remove_stopwords:
            filter_tokens = [word for word in filter_tokens if word not in self.stopwords]
        if to_lower:
            filter_tokens = [word.lower() for word in filter_tokens]

        return filter_tokens
