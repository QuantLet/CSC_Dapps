import numpy as np
import pandas as pd
import keras
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import Dense, Input, GRU, LSTM 
from keras.layers import Bidirectional, Dropout, GlobalMaxPool1D 
from keras.layers import LSTM, GRU, GlobalAveragePooling1D
from keras.layers import Conv1D, GlobalMaxPooling1D, TimeDistributed
from keras.layers import Dense, Embedding, Input

from keras.models import Model, Sequential
from keras.optimizers import RMSprop
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import load_model

from keras.preprocessing import text, sequence
from keras import initializers as initializers, regularizers, constraints

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

from nltk import tokenize 
import nltk
nltk.download('punkt')

#############################################
#####           ATTENTION              ######
#############################################

# Code for attention is based on the following implementation https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

# Code for attention is based on the following implementation https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

#############################################
#####              MODELS              ######
#############################################  

def gru_keras(max_features, maxlen, bidirectional, dropout_rate, embed_dim, rec_units,mtype='GRU', reduction = None):
    
    if K.backend == 'tensorflow':        
        K.clear_session()
        
    input_layer     = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features, output_dim=embed_dim, trainable=True)(input_layer)
    x               = SpatialDropout1D(dropout_rate)(embedding_layer)
    
    if reduction:
        if mtype   == 'GRU':
            if bidirectional:
                x           = Bidirectional(GRU(units=rec_units, return_sequences=True))(x)
            else:
                x           = GRU(units=rec_units, return_sequences=True)(x)
        elif mtype == 'LSTM':
            if bidirectional:
                x           = Bidirectional(LSTM(units=rec_units, return_sequences=True))(x)
            else:
                x           = LSTM(units=rec_units, return_sequences=True)(x) 
        
        if reduction == 'average':
          x = GlobalAveragePooling1D()(x)
        elif reduction == 'maximum':
          x = GlobalMaxPool1D()(x)
        elif reduction == 'attention':
          x = AttentionWithContext()(x)
    else: 
        if mtype   == 'GRU':
            if bidirectional:
                x           = Bidirectional(GRU(units=rec_units, return_sequences=False))(x)
            else:
                x           = GRU(units=rec_units, return_sequences=False)(x)
        elif mtype == 'LSTM':
            if bidirectional:
                x           = Bidirectional(LSTM(units=rec_units, return_sequences=False))(x)
            else:
                x           = LSTM(units=rec_units, return_sequences=False)(x) 
                
    output_layer = Dense(10, activation="sigmoid")(x)
    model        = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['acc'])
    return model
# Code for hierarchical attention network is based on the following implementation https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py   
def make_hat(max_sent_len, max_sent_amount, max_features, embed_dim, rec_units, dropout_rate):

    sentence_input = Input(shape=(max_sent_len,), dtype='int32')
    embedded_sequences = Embedding(max_features+1, embed_dim, trainable=True)(sentence_input)
    embedded_sequences = SpatialDropout1D(dropout_rate)(embedded_sequences)
    l_lstm = Bidirectional(GRU(rec_units, return_sequences=True))(embedded_sequences)
    l_att = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, l_att)

    comment_input = Input(shape=(max_sent_amount, max_sent_len), dtype='int32')
    comment_encoder = TimeDistributed(sentEncoder)(comment_input)
    l_lstm_sent = Bidirectional(GRU(rec_units, return_sequences=True))(comment_encoder)
    l_att_sent = AttentionWithContext()(l_lstm_sent)

    preds = Dense(10, activation='sigmoid')(l_att_sent)
    model = Model(comment_input, preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['acc'])
    return model
    
def cnn_keras(max_features, maxlen, dropout_rate, embed_dim, num_filters=300):
    if K.backend == 'tensorflow':        
        K.clear_session()
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features, output_dim=embed_dim, trainable=True)(input_layer)
    x = SpatialDropout1D(dropout_rate)(embedding_layer)
    x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    output_layer = Dense(10, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['acc'])
    return model

def dl_model(model_type='BGRU', max_features=40000, embed_dim=50, rec_units=150, dropout_rate=0.25, maxlen=400, max_sent_len=100, max_sent_amount=4):
                
    if model_type == 'GRU':
        return gru_keras(max_features=max_features, maxlen=maxlen, bidirectional=False, mtype='GRU', 
                         dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units)
    if model_type == 'LSTM':
        return gru_keras(max_features=max_features, maxlen=maxlen, bidirectional=False, mtype='LSTM', 
                         dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units)
    if model_type == 'BGRU':
        return gru_keras(max_features=max_features, maxlen=maxlen, bidirectional=True, mtype='GRU', 
                         dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units)
    if model_type == 'BLSTM':
        return gru_keras(max_features=max_features, maxlen=maxlen, bidirectional=True, mtype='LSTM', 
                         dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units)
    if model_type == 'BGRU_avg':
        return gru_keras(max_features=max_features, maxlen=maxlen, bidirectional=True, mtype='GRU', 
                         dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units, 
                         reduction='average')
    if model_type == 'BGRU_max':
        return gru_keras(max_features=max_features, maxlen=maxlen, bidirectional=True, mtype='GRU', 
                         dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units, 
                         reduction='maximum')
    if model_type == 'BGRU_att':
        return gru_keras(max_features=max_features, maxlen=maxlen, bidirectional=True, mtype='GRU', 
                         dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units, 
                         reduction='attention')
    if model_type == 'CNN': 
        return cnn_keras(max_features=max_features, maxlen=maxlen, dropout_rate=dropout_rate, embed_dim=embed_dim)
    if model_type == 'HAN': 
        return make_hat(max_sent_len=max_sent_len, max_sent_amount=max_sent_amount, max_features=max_features, embed_dim=embed_dim, rec_units=rec_units,dropout_rate=dropout_rate)
    if model_type == 'psHAN': 
        return make_hat(max_sent_len=max_sent_len, max_sent_amount=max_sent_amount, max_features=max_features, embed_dim=embed_dim, rec_units=rec_units, dropout_rate=dropout_rate)
        
#############################################
#####            TRAINING              ######
#############################################

def train_model(X, y,  mtype, cv,  
                epochs, cv_models_path, train, X_test=None, nfolds=None,
                y_test=None, rs=42, max_features=40000, maxlen=400, 
                dropout_rate=0.25, rec_units=150, embed_dim=50, 
                batch_size=256, max_sen_len=100, max_sent_amount=4,
                threshold=0.3):
    if cv:
        kf = StratifiedKFold(n_splits=nfolds, random_state=rs)
        auc = []
        roc = []
        fscore_ = [] 

        for c, (train_index, val_index) in enumerate(kf.split(X, y)):
            
            print(f' fold {c}')
            
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index] 
            
            tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)
            tokenizer.fit_on_texts(X_train)
            
            if mtype == 'HAN':
                def clean_str(string):
                    #string = string.replace(",", ".").replace(";", ".").replace(":", ".").replace("-", ".")
                    return string.strip().lower()
                
                def tok_sentence(s):
                    temp = tokenizer.texts_to_sequences(s)
                    if len(temp)==0:
                        return np.array([0])
                    return temp
                    
                    
                train_posts = []
                train_labels = []
                train_texts = []
                
                #TRAIN
                for i, value in enumerate(X_train):
                    if(i%10000==0):
                        print(i)
                    text = clean_str(value)
                    train_texts.append(text)
                    sentences = tokenize.sent_tokenize(text)
                    sentences = tok_sentence(sentences)
                    x = len(sentences)<max_sent_amount
                    while x:
                        sentences.append(np.array([0])) 
                        x = len(sentences)<max_sent_amount
            
                    if len(sentences)>max_sent_amount:
                        sentences = sentences[0:max_sent_amount]
                    sentences = sequence.pad_sequences(sentences, maxlen=max_sen_len)
            
                    train_posts.append(sentences)
                
                val_posts = []
                val_labels = []
                val_texts = []
            
                #VAL
                for i, value in enumerate(X_val):
                    if(i%10000==0):
                        print(i)
                    text = clean_str(value)
                    val_texts.append(text)
                    sentences = tokenize.sent_tokenize(text)
                    sentences = tok_sentence(sentences)
            
            
                    x = len(sentences)<max_sent_amount
                    while x:
                        sentences.append(np.array([0])) 
                        x = len(sentences)<max_sent_amount
            
                    if len(sentences)>max_sent_amount:
                        sentences = sentences[0:max_sent_amount]
                    sentences = sequence.pad_sequences(sentences, maxlen=max_sen_len)
                    val_posts.append(sentences)
                
                X_train = np.array(train_posts)
                y_train = np.array(y_train)
                X_val =  np.array(val_posts)
                y_val = np.array(y_val)
                
                del train_posts
                del val_posts
            elif mtype =='psHAN':
                X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_sen_len*max_sent_amount)
                X_val = sequence.pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_sen_len*max_sent_amount)
                X_train = np.array([line.reshape(max_sent_amount,max_sen_len) for line in X_train])
                X_val = np.array([line.reshape(max_sent_amount,max_sen_len) for line in X_val])
            else:
                list_tokenized_train = tokenizer.texts_to_sequences(X_train)
                list_tokenized_val   = tokenizer.texts_to_sequences(X_val)
                
                X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
                X_val   = sequence.pad_sequences(list_tokenized_val, maxlen=maxlen)
            
            model = dl_model(model_type=mtype, max_features=max_features, maxlen=maxlen, 
                            dropout_rate=dropout_rate, embed_dim=embed_dim, rec_units=rec_units,
                            max_sent_len=max_sen_len, max_sent_amount=max_sent_amount)
            
            print('Fitting')
            if train:
                model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)
                model.save_weights(f'{cv_models_path}/{mtype}_fold_{c}.h5')
            else: 
                model.load_weights(f'{cv_models_path}/{mtype}_fold_{c}.h5')
            
            probs = model.predict(X_val, batch_size=batch_size, verbose=1)
            
            #for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            threshold = threshold
            probs_class = probs.copy()
            probs_class[probs_class >= threshold] = 1 
            probs_class[probs_class < threshold] = 0
            precision = precision_score(y_val, probs_class) 
            recall    = recall_score(y_val, probs_class)
            fscore    = f1_score(y_val, probs_class)
            print(f' {threshold} fold {c} precision {round(precision, 3)} recall {round(recall, 3)} fscore {round(fscore,3)}')
            
            auc_f = average_precision_score(y_val, probs)
            
            auc.append(auc_f)
            roc_f = roc_auc_score(y_val, probs)
            roc.append(roc_f)
            fscore_.append(fscore)
            print(f'fold {c} average precision {round(auc_f, 3)}')
            print(f'fold {c} roc auc {round(roc_f, 3)}')
            
            del model
            K.clear_session()
        
        print(f'PR-C {round(np.array(auc).mean(), 3)}')
        print(f'ROC AUC {round(np.array(roc).mean(), 3)}')
        print(f'FScore {round(np.array(fscore_).mean(), 3)}')
        
        print(f'PR-C std {round(np.array(auc).std(), 3)}')
        print(f'ROC AUC std {round(np.array(roc).std(), 3)}')
        print(f'FScore std {round(np.array(fscore_).std(), 3)}')
    else:
            X_train   = X
            y_train   = y
            tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features, oov_token='unknown')
            tokenizer.fit_on_texts(X_train)
            
            
            if mtype == 'HAN':
                
                def clean_str(string):
                    #string = string.replace(",", ".").replace(";", ".").replace(":", ".").replace("-", ".")
                    return string.strip().lower()
                
                def tok_sentence(s):
                    temp = tokenizer.texts_to_sequences(s)
                    if len(temp)==0:
                        return np.array([0])
                    return temp
                
                train_posts = []
                train_labels = []
                train_texts = []
                
                # FULL TRAIN
                for i, value in enumerate(X):
                    if(i%10000==0):
                        print(i)
                    text = clean_str(value)
                    train_texts.append(text)
                    sentences = tokenize.sent_tokenize(text)
                    sentences = tok_sentence(sentences)
                    x = len(sentences)<max_sent_amount
                    while x:
                        sentences.append(np.array([0])) 
                        x = len(sentences)<max_sent_amount
                
                    if len(sentences)>max_sent_amount:
                        sentences = sentences[0:max_sent_amount]
                    sentences = sequence.pad_sequences(sentences, maxlen=max_sen_len)
                
                    train_posts.append(sentences)
                
                    
                test_posts = []
                test_labels = []
                test_texts = []
                    
                    
                #Test
                for i, value in enumerate(X_test):
                    if(i%10000==0):
                        print(i)
                    text = clean_str(value)
                    test_texts.append(text)
                    sentences = tokenize.sent_tokenize(text)
                    sentences = tok_sentence(sentences)
                    x = len(sentences)<max_sent_amount
                    while x:
                        sentences.append(np.array([0])) 
                        x = len(sentences)<max_sent_amount
                
                    if len(sentences)>max_sent_amount:
                        sentences = sentences[0:max_sent_amount]
                    sentences = sequence.pad_sequences(sentences, maxlen=max_sen_len)
                
                    test_posts.append(sentences)
                    
                    
                X_train = np.array(train_posts)
                y_train = np.array(y)
                X_test =  np.array(test_posts)
                y_test = np.array(y_test)
                
                del train_posts
                del test_posts
            elif mtype =='psHAN':
                X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_sen_len*max_sent_amount, padding='post')
                X_test  = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_sen_len*max_sent_amount, padding='post')
                X_train = np.array([line.reshape(max_sent_amount, max_sen_len) for line in X_train])
                X_test  = np.array([line.reshape(max_sent_amount, max_sen_len) for line in X_test])
            else:
                list_tokenized_train = tokenizer.texts_to_sequences(X_train)
                list_tokenized_test  = tokenizer.texts_to_sequences(X_test)
                X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
                X_test  = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
                
            y_train = np.array(y_train)
            y_test  = np.array(y_test)

            model = dl_model(model_type=mtype, max_features=max_features, 
                            maxlen=maxlen, dropout_rate=dropout_rate, embed_dim=embed_dim, 
                            rec_units=rec_units, max_sent_len=max_sen_len, max_sent_amount=max_sent_amount)
            
            print('Fitting')

            if train:
                model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)
                model.save_weights(f'{cv_models_path}/{mtype}.h5')
            else: 
                model.load_weights(f'{cv_models_path}/{mtype}.h5')
            probs = model.predict(X_test, batch_size=batch_size, verbose=1)
            auc_f = average_precision_score(y_test, probs)
            roc_f = roc_auc_score(y_test, probs)
            
            
            threshold = threshold
            probs_class = probs.copy()
            probs_class[probs_class >= threshold] = 1 
            probs_class[probs_class < threshold] = 0
            precision = precision_score(y_test, probs_class) 
            recall    = recall_score(y_test, probs_class)
            fscore    = f1_score(y_test, probs_class)
            
            print('_________________________________')
            print(f'PR-C is {round(auc_f,3)}')
            print('_________________________________\n')
            
            print('_________________________________')
            print(f'ROC AUC is {round(roc_f,3)}')
            print('_________________________________')
            
            print('_________________________________')
            print(f'FScore is {round(fscore,3)}')
            print('_________________________________\n')