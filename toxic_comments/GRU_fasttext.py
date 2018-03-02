#_*_coding:utf-8_*_

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import model
from keras.layers import Input,Dense,Embedding,SpatialDropout1D,concatenate
from keras.layers import GRU,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.preprocessing import text,sequence

from keras.callbacks imoprt Callback

import warnings
warnings.filterwarnings('ignore')

import os 
os.environ['OMP_NUM_THREADS'] = '4'

EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge')