#_*_coding:utf-8_*_
import sys
sys.path.append('..')
#sys.path.append('../kaggle_methods')
from dm_methods.kaggle_methods.ridge import ridge_cv
from dm_methods.kaggle_methods.xgboost_classification import xgboost_classification_cv
from dm_methods.kaggle_methods.logistic_regression import LogisticRegression_CV
from dm_methods.kaggle_methods.svc import SVC_CV
from dm_methods.kaggle_methods.nb_classification import GaussianNB_CV
from dm_methods.kaggle_methods.random_forest_classification import RandomForest_CV
from dm_methods.kaggle_methods.lightgbm_classification import lightgbm_CV
#from ridge import ridge_cv
from sklearn import metrics
import re
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import gc
from contextlib import contextmanager
import string
from scipy.sparse import hstack 

from sklearn.feature_extraction.text import TfidfVectorizer
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start




@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(name+' done in {} s'.format(time.time()-t0))


#处理数据
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df_train = pd.read_csv('./input/train.csv').fillna(" ")
df_test = pd.read_csv('./input/test.csv').fillna(" ")

df_train = df_train.loc[:1000,:]
df_test = df_test.loc[:1000,:]


label_name = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
train_label = df_train[label_name]
#df_all = pd.concat([df_train[['id','comment_text']],df_test],axis=0)

#add on 2018/3/5
# Contraction replacement patterns
cont_patterns = [
    ('(W|w)on\'t', 'will not'),
    ('(C|c)an\'t', 'can not'),
    ('(I|i)\'m', 'i am'),
    ('(A|a)in\'t', 'is not'),
    ('(\w+)\'ll', '\g<1> will'),
    ('(\w+)n\'t', '\g<1> not'),
    ('(\w+)\'ve', '\g<1> have'),
    ('(\w+)\'s', '\g<1> is'),
    ('(\w+)\'re', '\g<1> are'),
    ('(\w+)\'d', '\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = text
    # 2. Drop \n and  \t
    clean = clean.replace("\n", " ")
    clean = clean.replace("\t", " ")
    clean = clean.replace("\b", " ")
    clean = clean.replace("\r", " ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile('[%s]' % re.escape(string.punctuation))
    clean = " ".join([exclude.sub('', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub("\d+", " ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub('\s+', ' ', clean)
    # Remove ending space if any
    clean = re.sub('\s+$', '', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(" ", "# #", clean)  # Replace space
    clean = "#" + clean + "#"  # add leading and trailing #

    return clean

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))
def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]
def get_indicators_and_clean_comments(df):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))

get_indicators_and_clean_comments(df_train)
get_indicators_and_clean_comments(df_test)

with timer("Creating numerical features"):
        num_features = [f_ for f_ in df_train.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address'] + label_name]

        skl = MinMaxScaler()
        train_num_features = csr_matrix(skl.fit_transform(df_train[num_features]))
        test_num_features = csr_matrix(skl.fit_transform(df_test[num_features]))

# Get TF-IDF features
train_text = df_train['clean_comment']
test_text = df_test['clean_comment']
all_text = pd.concat([train_text, test_text])
# First on real words
with timer("Tfidf on word"):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 2),
        max_features=20000)
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
del word_vectorizer
gc.collect()

with timer("Tfidf on char n_gram"):
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=char_analyzer,
            analyzer='word',
            ngram_range=(1, 1),
            max_features=50000)
        char_vectorizer.fit(all_text)
        train_char_features = char_vectorizer.transform(train_text)
        test_char_features = char_vectorizer.transform(test_text)

del char_vectorizer
gc.collect()
print((train_char_features > 0).sum(axis=1).max())
del train_text
del test_text
gc.collect()

 # Now stack TF IDF matrices
with timer("Staking matrices"):
    csr_trn = hstack(
        [
            train_char_features,
            train_word_features,
            train_num_features
        ]
    ).tocsr()
    # del train_word_features
    del train_num_features
    del train_char_features
    gc.collect()
    csr_sub = hstack(
        [
            test_char_features,
            test_word_features,
            test_num_features
        ]
    ).tocsr()
    # del test_word_features
    del test_num_features
    del test_char_features
    gc.collect()
submissions = pd.DataFrame.from_dict({'id': df_test['id']})
del df_test
gc.collect()
# now the training data 
training = csr_trn
testing = csr_sub

import lightgbm as lgb
from lightgbm import LGBMModel

##get label
#label = df_train['toxic']
#
#train_data = lgb.Dataset(training,label = label)
#test_data = lgb.Dataset(testing)
#
##setting parameters
#param = {'num_leaves':31, 'objective':'binary'}
#param['metric'] = 'auc'
#num_round = 10
#bst = lgb.train(param, train_data, num_boost_round=num_round)
#print(bst.current_iteration())
#param['num_round'] = 20
#bst2 = lgb.train(param,train_data)
#print(bst2.current_iteration())

metric = metrics.log_loss
scoring = 'roc_auc'
lgb_cls = lightgbm_CV(training.toarray(),df_train['toxic'],metric)
lgb_model = lgb_cls.cross_validation(scoring)


