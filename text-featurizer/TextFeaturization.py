from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import math
import logging
from datetime import datetime
import itertools
import operator
from pprint import pprint
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
import re


logging.basicConfig(file_name='text_feature_build.txt', level=logging.INFO, )

stop_words = list(text.ENGLISH_STOP_WORDS)
reg_split = re.compile(r'\s+|-')
reg_remove = re.compile(r'^(fy1\d|\d{1,2}\w{3,5}|\d+)$', re.IGNORECASE)


class TextFeaturizer(object):
    """
    After query the text features, we can use the top frequency words to build the bag of words and tf-idf features.
    It also provide the truncated SVD-PCA algorithm to reduce the dimension of the features.
    The TextFeaturizer has the method transform and fit, using scikit-learn style api.
    Now only support one text feature field featurizer
    """
    def __init__(self, text_feature_list,
                 feature_prefix,
                 num_words=None,
                 num_svd_bow=None,
                 num_svd_tif=None,
                 n_gram=None,
                 filter_stop=True,
                 min_df=None,
                 bow=True,
                 tfidf=True):
        '''
        :param text_feature_list: list of name of features, now only support the single feature
        :param num_words: number of top frequent words used in build text features, default 1000
        :param num_svd_bow: number of principle components from SVD result of bag of words features, default 35
        :param num_svd_tif: number of principle components from SVD result of ti-idf features, default 45
        :param n_gram: number of word gram included in building text features, default 3
        :param filter_stop: binary label, filter stop word or not, default True
        :param min_df: number of minimum appearance of words in the documents, default 2
        :param bow: binary label, including bag of words feature or not
        :param tfidf: binary label, including tf-idf features or not
        '''

        default_n_gram = 3
        default_num_words = 1000
        default_num_svd_bow = 35
        default_num_svd_tif = 45
        default_min_df = 2

        self.text_feature = text_feature_list
        self.feature_prefix = feature_prefix

        if n_gram:
            self.n_gram = n_gram
        else:
            self.n_gram = default_n_gram

        if num_words:
            self.num_words = num_words
        else:
            self.num_words = default_num_words

        if num_svd_bow:
            self.num_svd_bow = num_svd_bow
        else:
            self.num_svd_bow = default_num_svd_bow

        if num_svd_tif:
            self.num_svd_tif = num_svd_tif
        else:
            self.num_svd_tif = default_num_svd_tif

        if min_df:
            self.min_df = min_df
        else:
            self.min_df = default_min_df

        if bow:
            self.bow = True
        else:
            self.bow = False

        if tfidf:
            self.tfidf = True
        else:
            self.tfidf = False

        if filter_stop:
            self.cv = CountVectorizer(stop_words='english', max_features=self.num_words, min_df=self.min_df,
                                      analyzer="word")
            self.tiv = TfidfVectorizer(stop_words='english', max_features=self.num_words, min_df=self.min_df,
                                       ngram_range=(1, self.n_gram))
        else:
            self.cv = CountVectorizer(max_features=self.num_words, min_df=self.min_df,
                                      analyzer="word")
            self.tiv = TfidfVectorizer(max_features=self.num_words, min_df=self.min_df,
                                       ngram_range=(1, self.n_gram))
        self.cv_fit = None
        self.tiv_fit = None
        self.bow_tsvd_fit = None
        self.tif_tsvd_fit = None

    @staticmethod
    def __string_stem__(string):
        '''
        stem the string , including remove '(){}:', split by '-', remove short words and extra spaces
        :param string: input unicode type string
        :return: stemed string , if not unicode, return "null"
        '''
        if isinstance(string, unicode):
            # remove all ':'
            string = string.replace(":", "")
            # split on '-'
            string = string.replace("-", " ")
            # remove short words less than three characters
            short_word = re.compile(r'\W*\b\w{1,2}\b')
            string = short_word.sub('', string)
            # remove '()'
            string = string.replace("(", " ")
            string = string.replace(")", " ")
            # remove '{}'
            string = string.replace("{", " ")
            string = string.replace("}", " ")
            # transform to lower case
            string = string.lower()
            # remove extra spaces
            string = re.sub('\s+', ' ', string).strip()
            return string
        else:
            # if not unicode, return 'null'
            return "null"

    def __clean_df__(self, df_input, feature_list):
        '''
        using string_stem method to clean the text fields in the input dataframe
        :param df_input:  input dataframe containing text features with column name 'text'
        :param feature_list: list of column names in dataframe of text features
        :return: cleaned dataframe
        '''

        df = df_input.copy()

        # for feature in feature_list:
            #df[feature] = df[feature].map(lambda x: self.__string_stem__(x))

        df['text'] = df['text'].map(lambda x: self.__string_stem__(x))
        return df

    def __merge_text__(self, feature_list):
        '''
        not used now, combined all the text fields into column 'text'
        :param feature_list: list of name of text features
        :return: dataframe with new field 'text' merged all the text features
        '''
        index = 0
        df = self.input_df.copy()
        while index < len(feature_list):
            if index == 0:
                df['text'] = df[feature_list[index]]
            else:
                df['text'] = df['text'] + ' ' + df[feature_list[0]]
            index += 1

        self.df['text'] = df['text']

    @staticmethod
    def __build__bow_encoder__(cv, df):
        '''
        build bag of words feature through cleaned text features
        :param cv: input counter vector
        :param df: input training dataframe
        :return: trained counter vector and bag of words features from training data
        '''
        cv.fit(df['text'])
        cv_bow_features = cv.transform(df['text'])
        return cv, cv_bow_features

    @staticmethod
    def __build_tif_encoder__(tiv, df):
        '''
        build tf-idf features through cleaned text features
        :param tiv: input tf-idf building vector
        :param df: input training dataframe
        :return: trained ti-idf vector and tf-idf features from training data
        '''
        tiv.fit(df['text'])
        tiv_tif_features = tiv.transform(df['text'])
        return tiv, tiv_tif_features

    def fit(self, input_data):
        """
        api to fit the input_data with text features
        :param input_data: input dataframe with text features
        :return: dataframe for the training data with the principle components from bag of words features and tf-idf features
        drop the original text feature
        """
        # input data frame preprocessing
        df_input = input_data
        text_feature = self.text_feature
        # rename the text feature column as text
        df_input = df_input.rename(columns={text_feature[0]: 'text'})

        # self.__merge_text__(text_feature_list)
        # clean the text dataframe
        df = self.__clean_df__(df_input, text_feature)

        # feature names
        self.expanded_text_features = []

        # if building bag of words feature, then we do bag of words and SVD
        if self.bow:
            logging.info('bow fit start')
            cv = self.cv
            cv_fit, cv_bow_features = self.__build__bow_encoder__(cv, df)
            # built the bag of word featurizor
            self.cv_fit = cv_fit
            num_svd_bow = self.num_svd_bow
            # using truncated SVD for sparse matrix features
            logging.info('SVD of bow features fit start')
            tsvd = TruncatedSVD(n_components=num_svd_bow, random_state=2017)
            tsvd.fit(cv_bow_features)
            # built the svd for bag of words features
            self.bow_tsvd_fit = tsvd
            st_bow_tsvd = tsvd.fit_transform(cv_bow_features)
            logging.info('SVD of bow features fit finish')

            for i in range(st_bow_tsvd.shape[1]):
                df[self.feature_prefix + '-' + 'st_bow_tsvd'+str(i)] = st_bow_tsvd[:, i]
                self.expanded_text_features.append(self.feature_prefix + '-' + 'st_bow_tsvd'+str(i))
            logging.info('bow building fit finish')

        # if building the tf-idf features
        if self.tfidf:
            logging.info('tf-idf fit start')
            tiv = self.tiv
            tiv_fit, tiv_tif_features = self.__build__tif_encoder__(tiv, df)
            # build the tf-idf featurizor
            self.tiv_fit = tiv_fit
            num_svd_tif = self.num_svd_tif
            logging.info('SVD of tf-idf features fit start')
            tsvd = TruncatedSVD(n_components=num_svd_tif, random_state=2017)
            tsvd.fit(tiv_tif_features)
            # build the svd for tf-idf features
            self.tif_tsvd_fit = tsvd
            st_tif_tsvd = tsvd.fit_transform(tiv_tif_features)
            logging.info('SVD of tf-idf features fit finish')

            for i in range(st_tif_tsvd.shape[1]):
                df[self.feature_prefix + '-' + 'st_tif_tsvd' + str(i)] = st_tif_tsvd[:, i]
                self.expanded_text_features.append(self.feature_prefix + '-' + 'st_tif_tsvd' + str(i))
            logging.info('tf-idf fit finish')

        # drop the original text feature
        df = df.drop('text', axis=1, inplace=False)

        return df

    def transform(self, test_data):
        '''
        transform the test data using the built bow, tf-idf and svd from training data
        :param test_data: input dataframe of test dataset
        :return:  dataframe for the test data with the principle components from bag of words features and tf-idf features
        drop the original text feature
        '''
        df_input = test_data
        text_feature = self.text_feature
        df_input = df_input.rename(columns={text_feature[0]: 'text'})

        # self.__merge_text__(text_feature_list)
        df = self.__clean_df__(df_input, text_feature)

        if self.bow:
            logging.info('bow transform start')
            cv_fit = self.cv_fit
            cv_bow_features = cv_fit.transform(df['text'])
            tsvd = self.bow_tsvd_fit
            logging.info('SVD of bow feature transform start')
            st_bow_tsvd = tsvd.transform(cv_bow_features)
            logging.info('SVD of bow feature transform finish')
            for i in range(st_bow_tsvd.shape[1]):
                df[self.feature_prefix + '-' + 'st_bow_tsvd' + str(i)] = st_bow_tsvd[:, i]
            logging.info('bow transform finish')

        if self.tfidf:
            logging.info('tf-idf transform start')
            tiv_fit = self.tiv_fit
            tiv_tif_features = tiv_fit.transform(df['text'])
            tsvd = self.tif_tsvd_fit
            logging.info('SVD of tf-idf feature transform start')
            st_tif_tsvd = tsvd.transform(tiv_tif_features)
            logging.info('SVD of tf-idf feature transform finish')
            for i in range(st_tif_tsvd.shape[1]):
                df[self.feature_prefix + '-' + 'st_tif_tsvd' + str(i)] = st_tif_tsvd[:, i]
            logging.info('tf-idf transform finish')

        df = df.drop('text', axis=1, inplace=False)
        return df
    
    def extract_and_join_text_features(self, test_data):
        return self.transform(test_data)




















