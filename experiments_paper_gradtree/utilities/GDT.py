import numpy as np
import random

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from typing import Callable
from focal_loss import SparseCategoricalFocalLoss

from livelossplot import PlotLosses

import os
import gc
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from IPython.display import Image
from IPython.display import display, clear_output

import pandas as pd
import sys

import warnings
warnings.filterwarnings('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#os.environ["PYTHONWARNINGS"] = "default"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

import tensorflow as tf
import tensorflow_addons as tfa

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
#tf.get_logger().setLevel('WARNING')
#tf.autograph.set_verbosity(1)

np.seterr(all="ignore")

#from keras import backend as K

import seaborn as sns
sns.set_style("darkgrid")

import time
import random

from utilities.utilities_GDT import *
from utilities.GDT import *
from utilities.DNDT import *

from joblib import Parallel, delayed

from itertools import product
from collections.abc import Iterable

from copy import deepcopy

from tensorflow.data import AUTOTUNE

import graphviz

def make_batch(iterable, n=1, random_seed=42):
    tf.random.set_seed(random_seed)
    iterable = tf.random.shuffle(value=iterable, seed=random_seed)
    #rng = np.random.default_rng(seed=random_seed)
    #rng.shuffle(iterable)
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def my_gather_nd(params, indices):
    idx_shape = tf.shape(indices)
    params_shape = tf.shape(params)
    idx_dims = idx_shape[-1]
    gather_shape = params_shape[idx_dims:]
    params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
    axis_step = tf.cast(tf.math.cumprod(params_shape[:idx_dims], exclusive=True, reverse=True), tf.int64)
    indices_flat = tf.reduce_sum(indices * axis_step, axis=-1)
    result_flat = tf.gather(params_flat, indices_flat)
    return tf.reshape(result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0))

def make_batch_det(iterable, n=1):
    l = iterable.shape[0]
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def sigmoid(x, factor=1, shift_horizontal=0.0, shift_vertical=0.0):
    x = 1/(1+K.exp(-factor*(x-shift_horizontal))) + shift_vertical
    return x

def tanh(x, factor=1, shift_horizontal=0, shift_vertical=0):
    x = (K.exp(factor*(x-shift_horizontal))-K.exp(-factor*(x-shift_horizontal)))/(K.exp(factor*(x-shift_horizontal))+K.exp(-factor*(x-shift_horizontal))) + shift_vertical
    return x

def poly1_cross_entropy(number_of_classes, epsilon=1.0, base_loss='crossentropy', focalLossGamma=2):
    def _poly1_cross_entropy(y_true, y_pred):

        y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
        y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)

        # pt, CE, and Poly1 have shape [batch].
        if base_loss == 'crossentropy':
            if number_of_classes > 2:
                loss_function = tf.keras.losses.get('categorical_crossentropy')
            else:
                loss_function = tf.keras.losses.get('binary_crossentropy')
        elif base_loss == 'binary_focal_crossentropy':
            if base_loss == 'binary_focal_crossentropy':
                if number_of_classes == 2:
                    loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=focalLossGamma)
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=focalLossGamma)
        else:
            loss_function = tf.keras.losses.get(base_loss)

        if number_of_classes > 2:
            pt = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.nn.softmax(y_pred), axis=-1)
        else:
            pt = tf.reduce_sum(tf.cast(tf.one_hot(tf.cast(tf.round(y_true), tf.int64), depth=number_of_classes), tf.float32) * tf.stack([1-tf.math.sigmoid(y_pred), tf.math.sigmoid(y_pred)], axis=1), axis=-1)

        loss_raw = loss_function(y_true, y_pred)

        Poly1 = loss_raw + epsilon * (1 - pt)
        loss = tf.reduce_mean(Poly1)
        return loss
    return _poly1_cross_entropy


class GDT(tf.Module):

    def __init__(
            self,
            number_of_variables,
            number_of_classes,

            objective,

            loss,

            focalLossGamma = 2,

            depth = 3,

            learning_rate_index = 1e-3,
            learning_rate_values = 1e-3,
            learning_rate_leaf = 1e-3,

            optimizer = 'adam',

            dropout = 0.0,

            split_index_activation_beta = 1.0,

            split_index_activation = 'entmax',#'softmax',

            output_activation = 'softmax',

            initializer = 'GlorotUniform',

            normalize = None,

            polyLoss = False,
            polyLossEpsilon = None,
        
            prunePostHoc = True,
            prune_threshold = 1,

            random_seed = 42,
            verbosity = 1):


        self.depth = depth

        self.objective = objective

        self.normalize = normalize

        self.learning_rate_index = learning_rate_index
        self.learning_rate_values = learning_rate_values
        self.learning_rate_leaf = learning_rate_leaf

        self.optimizer = optimizer
        self.dropout = dropout

        self.split_index_activation_beta = split_index_activation_beta
        self.split_index_activation = split_index_activation
        self.output_activation = output_activation

        self.initializer = initializer

        self.seed = random_seed
        self.verbosity = verbosity
        self.number_of_variables = number_of_variables
        self.number_of_classes = number_of_classes

        self.prunePostHoc = prunePostHoc
        self.prune_threshold = prune_threshold
        
        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth


        self.loss_name = loss
        self.focalLossGamma = focalLossGamma
        self.polyLoss = polyLoss
        self.polyLossEpsilon = polyLossEpsilon

        if self.polyLoss:
            self.loss = poly1_cross_entropy(self.number_of_classes, epsilon=self.polyLossEpsilon, base_loss=loss, focalLossGamma=self.focalLossGamma)
        else:
            if self.loss_name == 'crossentropy':
                if self.number_of_classes == 2:
                    self.loss = tf.keras.losses.get('binary_crossentropy')
                else:
                    self.loss = tf.keras.losses.get('categorical_crossentropy')
            elif self.loss_name == 'binary_focal_crossentropy':
                if self.number_of_classes == 2:
                    self.loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=self.focalLossGamma)
                else:
                    self.loss = SparseCategoricalFocalLoss(gamma=self.focalLossGamma)
            else:
                self.loss = tf.keras.losses.get(self.loss_name)

        leaf_classes_array_shape = (self.leaf_node_num_,) if self.number_of_classes == 2 or self.objective == 'regression' else(self.leaf_node_num_, self.number_of_classes)

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.keras.utils.set_random_seed(int(self.seed))

        if self.initializer == 'dataAware':
            self.split_values = tf.Variable(tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=self.seed)(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_values', dtype=tf.float32)

            self.split_index_array = tf.Variable(tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=self.seed)(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_index_array', dtype=tf.float32)

            self.leaf_classes_array = tf.Variable(tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=self.seed)(shape=leaf_classes_array_shape, dtype=tf.float32),trainable=True,name='leaf_classes_array', dtype=tf.float32)
        else:
            self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_values', dtype=tf.float32)

            self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_index_array', dtype=tf.float32)

            self.leaf_classes_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=leaf_classes_array_shape, dtype=tf.float32),trainable=True,name='leaf_classes_array', dtype=tf.float32)
           
        if self.optimizer == 'QHAdam':
            self.optimizer_tree_split_index_array = QHAdamOptimizer(learning_rate=self.learning_rate_index)
            self.optimizer_tree_split_values = QHAdamOptimizer(learning_rate=self.learning_rate_values)
            self.optimizer_tree_leaf_classes_array = QHAdamOptimizer(learning_rate=self.learning_rate_leaf)
        elif self.optimizer == 'SWA':
            self.optimizer_tree_split_index_array = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_index), average_period=5)
            self.optimizer_tree_split_values = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_values), average_period=5)
            self.optimizer_tree_leaf_classes_array = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_leaf), average_period=5)            
        else:
            self.optimizer_tree_split_index_array = tf.keras.optimizers.get(self.optimizer)
            self.optimizer_tree_split_values = tf.keras.optimizers.get(self.optimizer)
            self.optimizer_tree_leaf_classes_array = tf.keras.optimizers.get(self.optimizer)

            self.optimizer_tree_split_index_array.learning_rate = self.learning_rate_index
            self.optimizer_tree_split_values.learning_rate = self.learning_rate_values
            self.optimizer_tree_leaf_classes_array.learning_rate = self.learning_rate_leaf


        self.plotlosses = PlotLosses()



    def forward(self,
                X: tf.Tensor,
                training: bool):
        X = tf.dtypes.cast(tf.convert_to_tensor(X), tf.float32)

        split_index_array_complete = self.split_index_array
        split_values_complete = self.split_values

        split_index_array_complete = self.apply_dropout(split_index_array_complete,
                                                        training=training)

        if self.split_index_activation == 'softmax':
            split_index_array_complete = tf.keras.activations.softmax(self.split_index_activation_beta * split_index_array_complete)
        elif self.split_index_activation == 'entmax':
            split_index_array_complete = entmax15(self.split_index_activation_beta * split_index_array_complete)
        elif self.split_index_activation == 'sparsemax':
            split_index_array_complete = tfa.activations.sparsemax(self.split_index_activation_beta * split_index_array_complete)

        adjust_constant = tf.stop_gradient(split_index_array_complete -  tfa.seq2seq.hardmax(split_index_array_complete))
        split_index_array_complete = split_index_array_complete - adjust_constant

        split_index_array_complete_selected = tf.squeeze(my_gather_nd(split_index_array_complete, indices=tf.reshape(self.internal_node_index_list, (-1,1))))
        split_values_complete_selected = tf.squeeze(my_gather_nd(split_values_complete, indices=tf.reshape(self.internal_node_index_list, (-1,1))))

        split_index_array_complete_selected = tf.reshape(split_index_array_complete_selected, (self.leaf_node_num_,self.depth,self.number_of_variables))
        split_values_complete_selected = tf.reshape(split_values_complete_selected, (self.leaf_node_num_,self.depth,self.number_of_variables))


        s1_sum = tf.einsum("ijk,ijk->ij", split_values_complete_selected, split_index_array_complete_selected)
        s2_sum = tf.einsum("ik,jlk->ijl", X, split_index_array_complete_selected)
        node_result = tf.sigmoid(s1_sum-s2_sum)
        node_result_corrected = node_result - tf.stop_gradient(node_result - tf.round(node_result))

        p = tf.reduce_prod(((1-self.path_identifier_list)*node_result_corrected + self.path_identifier_list*(1-node_result_corrected)), axis=2)

        if self.objective == 'regression' or self.number_of_classes == 2:
            function_values_gdt = tf.einsum('i,ji->j', self.leaf_classes_array, p) 
        else:
            function_values_gdt = tf.einsum('ij,ki->kj', self.leaf_classes_array, p)

        return function_values_gdt

    def predict(self, X, batch_size = 512, return_proba=False, denormalize=True):
        X = tf.dtypes.cast(tf.convert_to_tensor(X), tf.float32)

        if self.number_of_classes == 2:
            preds = tf.constant([], dtype=tf.float32)
        else:
            preds = tf.constant([], shape=(0,self.number_of_classes), dtype=tf.float32)

        for i, X_batch in enumerate(make_batch_det(X, batch_size)):
            preds_batch = self.forward_tf_function(X_batch,
                                                  training=False)

            if self.objective == 'classification':
                if self.number_of_classes == 2:
                    if return_proba:
                        preds_batch = tf.math.sigmoid(preds_batch)
                    else:
                        preds_batch = tf.round(tf.math.sigmoid(preds_batch))
                else:
                    if self.output_activation == "entmax":
                        if return_proba:
                            preds_batch = entmax15(preds_batch)
                        else:
                            preds_batch = tf.argmax(entmax15(preds_batch), axis=1)
                    elif self.output_activation == "sparsemax":
                        if return_proba:
                            preds_batch = tfa.activations.sparsemax(preds_batch)
                        else:
                            preds_batch = tf.argmax(tfa.activations.sparsemax(preds_batch), axis=1)
                    else: #self.output_activation == "softmax":
                        if return_proba:
                            preds_batch = tf.keras.activations.softmax(preds_batch)
                        else:
                            preds_batch = tf.argmax(tf.keras.activations.softmax(preds_batch), axis=1)

            elif self.objective == 'regression':
                if denormalize:
                    preds_batch = self.denormalize_labels(preds_batch)

            preds = tf.concat([preds, preds_batch], axis=0)

        return preds


    def test_step(self, X, y):

        predicted = self.forward(X,
                             training=False)

        if self.objective == 'classification':
            if self.number_of_classes == 2:
                current_loss = tf.reduce_mean(self.loss(y, tf.sigmoid(predicted)))
            else:
                if self.output_activation == "softmax":
                    current_loss = tf.reduce_mean(self.loss(y, tf.keras.activations.softmax(predicted)))
                elif self.output_activation == "entmax":
                    current_loss = tf.reduce_mean(self.loss(y, entmax15(predicted)))
                elif self.output_activation == "sparsemax":
                    current_loss = tf.reduce_mean(self.loss(y, tfa.activations.sparsemax(predicted)))
        else:
            current_loss = tf.reduce_mean(self.loss(y, predicted))

        if self.calculate_metric:
            if self.number_of_classes == 2:
                self.metric_val.update_state(tf.one_hot(tf.cast(tf.round(y), tf.int64), depth=self.number_of_classes, axis=-1),
                                            tf.stack([1-tf.math.sigmoid(predicted), tf.math.sigmoid(predicted)], axis=1))



            else:
                if self.number_of_classes >= 2 and self.loss_name == 'binary_focal_crossentropy':
                    self.metric_val.update_state(tf.one_hot(tf.cast(tf.round(tf.squeeze(y)), tf.int64), depth=self.number_of_classes, axis=-1),
                                                predicted)
                else:
                    self.metric_val.update_state(y,
                                                predicted)

        return current_loss





    
    #@tf.function(jit_compile=True)
    def forward_tf_function(self,
                            X: tf.Tensor,
                            training: bool):

        return self.forward(X=X,
                           training=training)

    def backward(self,
                 x: tf.Tensor,
                 y: tf.Tensor):


        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_tree_leaf_classes_array:
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_tree_split_values:
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_tree_split_index_array:
                    tape_tree_leaf_classes_array.watch(self.leaf_classes_array)
                    tape_tree_split_values.watch(self.split_values)
                    tape_tree_split_index_array.watch(self.split_index_array)

                    predicted = self.forward(x, training=True)

                    if self.objective == 'classification':
                        if self.number_of_classes == 2:
                            current_loss = tf.reduce_mean(self.loss(y, tf.sigmoid(predicted)))
                        else:
                            if self.output_activation == "softmax":
                                current_loss = tf.reduce_mean(self.loss(y, tf.keras.activations.softmax(predicted)))
                            elif self.output_activation == "entmax":
                                current_loss = tf.reduce_mean(self.loss(y, entmax15(predicted)))
                            elif self.output_activation == "sparsemax":
                                current_loss = tf.reduce_mean(self.loss(y, tfa.activations.sparsemax(predicted)))
                    else:
                        current_loss = tf.reduce_mean(self.loss(y, predicted))

        grads1 = tape_tree_leaf_classes_array.gradient(current_loss, self.leaf_classes_array)
        self.optimizer_tree_leaf_classes_array.apply_gradients(zip([grads1], [self.leaf_classes_array]))
        grads2 = tape_tree_split_values.gradient(current_loss, self.split_values)
        self.optimizer_tree_split_values.apply_gradients(zip([grads2], [self.split_values]))
        grads3 = tape_tree_split_index_array.gradient(current_loss, self.split_index_array)
        self.optimizer_tree_split_index_array.apply_gradients(zip([grads3], [self.split_index_array]))

        if self.calculate_metric:
            if self.number_of_classes == 2:
                self.metric.update_state(
                                    tf.one_hot(tf.cast(tf.round(y), tf.int32), depth=self.number_of_classes, axis=-1),
                                    tf.stack([1-tf.math.sigmoid(predicted), tf.math.sigmoid(predicted)], axis=1)
                )

            else:
                if self.number_of_classes >= 2 and self.loss_name == 'binary_focal_crossentropy':
                    self.metric.update_state(tf.one_hot(tf.cast(tf.round(tf.squeeze(y)), tf.int64), depth=self.number_of_classes, axis=-1),
                                            predicted)
                else:
                    self.metric.update_state(y,
                                            predicted)

        return current_loss#, predicted

    def get_params(self):
        return {
            'depth': self.depth,
            'normalize': self.normalize,
            'learning_rate_index': self.learning_rate_index,
            'learning_rate_leaf': self.learning_rate_leaf,
            'learning_rate_values': self.learning_rate_values,
            'optimize': self.optimizer,
            'dropout': self.dropout,
            'split_index_activation_beta': self.split_index_activation_beta,
            'split_index_activation': self.split_index_activation,
            'output_activation': self.output_activation,
            'initializer': self.initializer,
            
            'prunePostHoc': self.prunePostHoc,
            'prune_threshold': self.prune_threshold,

            'loss': self.loss_name,
            'polyLoss': self.polyLoss,
            'polyLossEpsilon': self.polyLossEpsilon,
            'split_index_activation_beta': self.split_index_activation_beta,

        }

    def train_loop(self, train_data):

        loss_list = tf.constant([])

        for index, (X_batch, y_batch) in enumerate(train_data):
            current_loss = self.backward_function(tf.dtypes.cast(tf.convert_to_tensor(X_batch), tf.float32), tf.dtypes.cast(tf.convert_to_tensor(y_batch), tf.float32))#self.backward(X_batch, y_batch)

            if self.verbosity > 2:
                batch_idx = (index+1)*self.batch_size
                msg = "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                print(msg.format(current_epoch, batch_idx, current_loss))

            loss_list = tf.concat([loss_list, [current_loss]], axis=0)

        return tf.reduce_mean(loss_list)

    def test_loop(self, valid_data):

        loss_list_val = tf.constant([])
        for index, (X_batch_valid, y_batch_valid) in enumerate(valid_data):
            current_loss_val = self.test_step_function(tf.dtypes.cast(tf.convert_to_tensor(X_batch_valid), tf.float32), tf.dtypes.cast(tf.convert_to_tensor(y_batch_valid), tf.float32))

            loss_list_val = tf.concat([loss_list_val, [current_loss_val]], axis=0)

        return tf.reduce_mean(loss_list_val)

    def fit(self,
            X_train,
            y_train,

            batch_size=512,
            epochs=1_000,

            restarts = 0,
            restart_type='loss',#'metric'

            early_stopping_epochs=50,
            early_stopping_type='loss',#'metric'
            early_stopping_epsilon = 0,

            valid_data=None,
            ):

        batch_size = min(batch_size, int(np.ceil(X_train.shape[0]/2)))
        self.batch_size = batch_size

        self.path_identifier_list = []
        self.internal_node_index_list = []
        for leaf_index in tf.unstack(tf.constant([i for i in range(self.leaf_node_num_)])):
            for current_depth in tf.unstack(tf.constant([i for i in range(1, self.depth+1)])):
                path_identifier = tf.cast(tf.math.floormod(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-current_depth)))), 2), tf.float32)
                internal_node_index =  tf.cast(tf.cast(tf.math.pow(2, (current_depth-1)), tf.float32) + tf.cast(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-(current_depth-1))))), tf.float32) - 1.0, tf.int64)
                self.path_identifier_list.append(path_identifier)
                self.internal_node_index_list.append(internal_node_index)
        self.path_identifier_list = tf.reshape(tf.stack(self.path_identifier_list), (-1,self.depth))
        self.internal_node_index_list = tf.reshape(tf.cast(tf.stack(self.internal_node_index_list), tf.int64), (-1,self.depth))

        X_train_numpy = X_train.values
        X_train = tf.dtypes.cast(tf.convert_to_tensor(X_train), tf.float32)
        y_train = tf.dtypes.cast(tf.convert_to_tensor(y_train), tf.float32)

        self.calculate_metric = True if self.verbosity == 1 or restart_type == 'metric' or early_stopping_type == 'metric' else False

        if self.calculate_metric:
            if self.objective == 'classification':
                if self.number_of_classes >= 2:
                    metric_name = 'f1'
                    self.metric = tfa.metrics.F1Score(average='macro', num_classes=self.number_of_classes, threshold=0.5)#tf.keras.metrics.CategoricalAccuracy()
                    self.metric_val = tfa.metrics.F1Score(average='macro', num_classes=self.number_of_classes, threshold=0.5)#tf.keras.metrics.CategoricalAccuracy()
                else:
                    metric_name = 'f1'
                    self.metric = tfa.metrics.F1Score(average='macro', num_classes=self.number_of_classes, threshold=0.5)#tf.keras.metrics.BinaryAccuracy()
                    self.metric_val = tfa.metrics.F1Score(average='macro', num_classes=self.number_of_classes, threshold=0.5)#tf.keras.metrics.BinaryAccuracy()
            elif self.objective == 'regression':
                metric_name = 'r2'
                self.metric = tfa.metrics.r_square.RSquare()
                self.metric_val = tfa.metrics.r_square.RSquare()

        if valid_data is not None:
            X_data_valid_numpy = valid_data[0].values
            valid_data = (tf.dtypes.cast(tf.convert_to_tensor(valid_data[0]), tf.float32),
                          tf.dtypes.cast(tf.convert_to_tensor(valid_data[1]), tf.float32))

        if self.normalize is not None:
            self.data_mean = tf.cast(tf.math.reduce_mean(y_train), tf.float32)
            self.data_std = tf.cast(tf.math.reduce_std(y_train), tf.float32)
            self.data_min = tf.cast(tf.math.reduce_min(y_train), tf.float32)
            self.data_max = tf.cast(tf.math.reduce_max(y_train), tf.float32)

        if self.objective == 'classification':
            if self.number_of_classes > 2 and (len(y_train.shape) == 1 or y_train.shape[1] == 1):
                if isinstance(y_train, pd.Series):
                    y_train = y_train.values
                #y_train = np_utils.to_categorical(tf.reshape(y_train, (-1,1)), num_classes=self.number_of_classes)
                if self.number_of_classes >= 2 and self.loss_name == 'binary_focal_crossentropy':
                    y_train = tf.reshape(y_train, (-1,1))
                else:
                    y_train = tf.one_hot(tf.cast(tf.round(y_train), tf.int64), depth=self.number_of_classes, axis=-1)
                if valid_data is not None:
                    valid_data_labels = valid_data[1]
                    if isinstance(valid_data_labels, pd.Series):
                        valid_data_labels = valid_data_labels.values
                    #valid_data_labels = np_utils.to_categorical(tf.reshape(valid_data_labels, (-1,1)), num_classes=self.number_of_classes)
                    if self.number_of_classes >= 2 and self.loss_name == 'binary_focal_crossentropy':
                        valid_data_labels = tf.reshape(valid_data_labels, (-1,1))
                    else:
                        valid_data_labels = tf.one_hot(tf.cast(tf.round(valid_data_labels), tf.int64), depth=self.number_of_classes, axis=-1)
                    valid_data = (valid_data[0], valid_data_labels)

        else:
            y_train = self.normalize_labels(y_train)

            if valid_data is not None:
                valid_data = (valid_data[0], self.normalize_labels(valid_data[1]))




        self.tfDataset = True if sys.getsizeof(X_train) > 5*1e8 else False #greater than 500MB


        if self.batch_size < valid_data[0].shape[0]:
            if self.tfDataset:
                train_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(X_train), tf.float32),  tf.dtypes.cast(tf.convert_to_tensor(y_train), tf.float32)))
                train_data = (train_data
                        .shuffle(32_768)
                        .cache()
                        #.repeat()
                        .batch(batch_size=batch_size) #, drop_remainder=True, num_parallel_calls=None
                        .prefetch(AUTOTUNE)      #AUTOTUNE
                             )

                valid_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(valid_data[0]), tf.float32),  tf.dtypes.cast(tf.convert_to_tensor(valid_data[1]), tf.float32)))
                valid_data = (valid_data
                        #.shuffle(32_768)
                        .cache()
                        #.repeat()
                        .batch(batch_size=batch_size) #, drop_remainder=True, num_parallel_calls=None
                        .prefetch(AUTOTUNE)
                             )
            else:
                valid_data = list(zip(make_batch_det(valid_data[0], self.batch_size), make_batch_det(valid_data[1], self.batch_size)))
        else:
            self.tfDataset = False
            valid_data = [[tf.constant(valid_data[0]),  tf.constant(valid_data[1])]]


        split_values_best_model = None#tf.identity(self.split_values)
        split_index_array_best_model = None#tf.identity(self.split_index_array)
        leaf_classes_array_best_model = None#tf.identity(self.leaf_classes_array)

        best_model_minimum_loss = np.inf
        best_model_minimum_metric = -np.inf

        disable = True if self.verbosity == -1 else False

        for restart_number in tqdm(range(restarts+1), desc='restarts', disable=disable):

            self.backward_function = tf.function(self.backward, jit_compile=True)
            self.train_loop_function = self.train_loop
            self.test_step_function = tf.function(self.test_step, jit_compile=True)
            self.test_loop_function = self.test_loop

            self.seed += restart_number

            if restart_number > 0:

                tf.keras.backend.clear_session()

                if self.optimizer == 'QHAdam':
                    self.optimizer_tree_split_index_array = QHAdamOptimizer(learning_rate=self.learning_rate_index)
                    self.optimizer_tree_split_values = QHAdamOptimizer(learning_rate=self.learning_rate_values)
                    self.optimizer_tree_leaf_classes_array = QHAdamOptimizer(learning_rate=self.learning_rate_leaf)
                elif self.optimizer == 'SWA':
                    self.optimizer_tree_split_index_array = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_index), average_period=5)
                    self.optimizer_tree_split_values = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_values), average_period=5)
                    self.optimizer_tree_leaf_classes_array = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_leaf), average_period=5)                   
                else:
                    self.optimizer_tree_split_index_array = tf.keras.optimizers.get(self.optimizer)
                    self.optimizer_tree_split_values = tf.keras.optimizers.get(self.optimizer)
                    self.optimizer_tree_leaf_classes_array = tf.keras.optimizers.get(self.optimizer)

                    self.optimizer_tree_split_index_array.learning_rate = self.learning_rate_index
                    self.optimizer_tree_split_values.learning_rate = self.learning_rate_values
                    self.optimizer_tree_leaf_classes_array.learning_rate = self.learning_rate_leaf

                tf.random.set_seed(self.seed)
                np.random.seed(self.seed)
                random.seed(self.seed)
                tf.keras.utils.set_random_seed(int(self.seed))
                #tf.config.experimental.enable_op_determinism()

                if self.initializer == 'dataAware':
                    self.split_values.assign(tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=self.seed)(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32))

                    self.split_index_array.assign(tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=self.seed)(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32))

                    self.leaf_classes_array.assign(tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=self.seed)(shape=leaf_classes_array_shape, dtype=tf.float32))
                else:
                    self.split_values.assign(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32))

                    #tf.random.set_seed(self.seed)
                    self.split_index_array.assign(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32))

                    leaf_classes_array_shape = (self.leaf_node_num_,) if self.number_of_classes == 2 or self.objective == 'regression' else(self.leaf_node_num_, self.number_of_classes)

                    #tf.random.set_seed(self.seed)
                    self.leaf_classes_array.assign(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=leaf_classes_array_shape, dtype=tf.float32))


            if self.initializer == 'dataAware':
                thresholds = []
                for internal_node_index in range(self.internal_node_num_):
                    threshold_by_variable = []
                    for variable_index in range(self.number_of_variables):
                        random_variable_value = np.random.choice(X_train[:,variable_index], 1)
                        threshold_by_variable.append(random_variable_value)

                    thresholds.append(tf.squeeze(tf.stack(threshold_by_variable)))

                thresholds = tf.stack(thresholds)

                self.split_values.assign(thresholds)


            if self.optimizer not in ['QHAdam', 'SWA']:
                self.optimizer_tree_leaf_classes_array.build([self.leaf_classes_array])
                self.optimizer_tree_split_values.build([self.split_values])
                self.optimizer_tree_split_index_array.build([self.split_index_array])


            minimum_loss_epoch = np.inf
            minimum_loss_epoch_valid = np.inf

            minimum_metric_epoch = -np.inf
            minimum_metric_epoch_valid = -np.inf

            epochs_without_improvement = 0

            for current_epoch in tqdm(range(epochs), desc='epochs', disable=disable):
                #tf.random.set_seed(self.seed + current_epoch)
                #X_train_epoch = tf.random.shuffle(X_train, seed=self.seed + current_epoch)
                #tf.random.set_seed(self.seed + current_epoch)
                #y_train_epoch = tf.random.shuffle(y_train, seed=self.seed + current_epoch)
                if self.batch_size < X_train.shape[0]:
                    if not self.tfDataset:
                        tf.random.set_seed(self.seed + current_epoch)
                        X_train_epoch = tf.random.shuffle(X_train, seed=self.seed + current_epoch)
                        tf.random.set_seed(self.seed + current_epoch)
                        y_train_epoch = tf.random.shuffle(y_train, seed=self.seed + current_epoch)
                            #train_data = list(zip(tf.ragged.constant(list(make_batch_det(X_train_epoch, self.batch_size))), tf.ragged.constant(list(make_batch_det(y_train_epoch, self.batch_size)))))
                        train_data = zip(make_batch_det(X_train_epoch, self.batch_size), make_batch_det(y_train_epoch, self.batch_size))
                else:
                    tf.random.set_seed(self.seed + current_epoch)
                    X_train_epoch = tf.random.shuffle(X_train, seed=self.seed + current_epoch)
                    tf.random.set_seed(self.seed + current_epoch)
                    y_train_epoch = tf.random.shuffle(y_train, seed=self.seed + current_epoch)

                    train_data = [[tf.constant(X_train_epoch),  tf.constant(y_train_epoch)]]

                #loss_list = self.train_loop(valid_data)
                #current_loss_epoch = tf.reduce_mean(loss_list)

                current_loss_epoch = self.train_loop_function(train_data)


                #tf.print('preds_list_logits', preds_list_logits)
                #preds_logits = np.nan_to_num(tf.concat(preds_list_logits, axis=0))

                loss_dict = {'loss': current_loss_epoch}
                if self.calculate_metric:
                    loss_dict[metric_name] = self.metric.result()
                    self.metric.reset_states()

                if valid_data is not None:
                    #loss_list_val = self.test_loop(train_data)
                    #current_loss_epoch_valid = tf.reduce_mean(loss_list_val)
                    current_loss_epoch_valid = self.test_loop_function(valid_data)

                    loss_dict['val_loss'] = current_loss_epoch_valid
                    if self.calculate_metric:
                        loss_dict['val_' + metric_name] = self.metric_val.result()
                        self.metric_val.reset_states()


                if self.verbosity > 1:
                    msg = "Epoch: {:02d} | Loss: {:.5f} |"
                    print(msg.format(current_epoch, current_loss_epoch))
                    if valid_data is not None:
                        msg = "Epoch: {:02d} | Valid Loss: {:.5f} |"
                        print(msg.format(current_epoch, current_loss_epoch_valid))

                if self.verbosity == 1:


                    self.plotlosses.update(loss_dict)
                    self.plotlosses.send()

                if early_stopping_type == 'metric' or restart_type == 'metric':

                    if valid_data is not None:
                        if loss_dict['val_' + metric_name] - early_stopping_epsilon > minimum_metric_epoch_valid:
                            minimum_metric_epoch_valid = loss_dict['val_' + metric_name]#current_loss_epoch_valid
                            if early_stopping_type == 'metric':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)

                        else:
                            if early_stopping_type == 'metric':
                                epochs_without_improvement += 1
                    else:
                        if loss_dict[metric_name] - early_stopping_epsilon > minimum_metric_epoch:
                            minimum_metric_epoch = loss_dict[metric_name]#current_loss_epoch

                            if early_stopping_type == 'metric':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)

                        else:
                            if early_stopping_type == 'metric':
                                epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_epochs:
                        try:
                            self.split_values.assign(split_values_stored)
                            self.split_index_array.assign(split_index_array_stored)
                            self.leaf_classes_array.assign(leaf_classes_array_stored)
                        except UnboundLocalError:
                            pass

                        break
                if early_stopping_type == 'loss' or restart_type == 'loss':
                    if valid_data is not None:
                        if current_loss_epoch_valid + early_stopping_epsilon < minimum_loss_epoch_valid:
                            minimum_loss_epoch_valid = current_loss_epoch_valid

                            if early_stopping_type == 'loss':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)

                        else:
                            if early_stopping_type == 'loss':
                                epochs_without_improvement += 1
                    else:
                        if current_loss_epoch + early_stopping_epsilon < minimum_loss_epoch:
                            minimum_loss_epoch = current_loss_epoch

                            if early_stopping_type == 'loss':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)

                        else:
                            if early_stopping_type == 'loss':
                                epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_epochs:
                        try:
                            self.split_values.assign(split_values_stored)
                            self.split_index_array.assign(split_index_array_stored)
                            self.leaf_classes_array.assign(leaf_classes_array_stored)
                        except UnboundLocalError:
                            pass
                        break


            if valid_data is not None:
                if restart_type == 'metric':
                    if minimum_metric_epoch_valid > best_model_minimum_metric or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:

                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)

                        best_model_minimum_metric = minimum_metric_epoch_valid

                else:
                    if minimum_loss_epoch_valid < best_model_minimum_loss or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:

                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)

                        best_model_minimum_loss = minimum_loss_epoch_valid
            else:
                if restart_type == 'metric':
                    if minimum_metric_epoch > best_model_minimum_metric or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)

                        best_model_minimum_metric = minimum_metric_epoch

                else:
                    if minimum_loss_epoch < best_model_minimum_loss or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)

                        best_model_minimum_loss = minimum_loss_epoch


        del self.backward_function
        del self.test_step_function
        del self.train_loop_function
        del self.test_loop_function

        if self.calculate_metric:
            del self.metric_val
            del self.metric

        del self.optimizer_tree_split_index_array
        del self.optimizer_tree_split_values
        del self.optimizer_tree_leaf_classes_array

        try:
            self.split_values.assign(split_values_best_model)
            self.split_index_array.assign(split_index_array_best_model)
            self.leaf_classes_array.assign(leaf_classes_array_best_model)
        except UnboundLocalError:
            pass
            
        
        if self.prunePostHoc:
            split_values = self.split_values.numpy()
            leaf_classes_array = self.leaf_classes_array.numpy()#tf.sigmoid(self.leaf_classes_array).numpy()
            split_index_array = tf.cast(tfa.seq2seq.hardmax(self.split_index_array), tf.int64).numpy()
            
            tree = DecisionTree(split_values, split_index_array, leaf_classes_array)

            tree.prune_tree(data=X_train_numpy, min_samples=self.prune_threshold)          
            
            split_values_pruned, split_index_array_pruned, leaf_classes_array_pruned = tree.to_array_representation()
            
            self.split_values.assign(split_values_pruned)
            self.split_index_array.assign(split_index_array_pruned)
            self.leaf_classes_array.assign(leaf_classes_array_pruned)  
            
            self.num_internal_nodes_acutal, self.num_leaf_nodes_acutal = tree.count_nodes()
     
            self.tree_class = tree
    
        else:
            self.num_leaf_nodes_acutal = self.leaf_node_num_
            self.num_internal_nodes_acutal = self.internal_node_num_
            
            split_values = self.split_values.numpy()
            leaf_classes_array = self.leaf_classes_array.numpy()#tf.sigmoid(self.leaf_classes_array).numpy()
            split_index_array = tf.cast(tfa.seq2seq.hardmax(self.split_index_array), tf.int64).numpy()
            
            tree = DecisionTree(split_values, split_index_array, leaf_classes_array)
            self.tree_class = tree
            
    def set_params(self, **kwargs):

        #print(kwargs)

        conditional_arguments = ['random_seed',
                                'depth',
                                'initializer',
                                'optimizer',
                                'learning_rate_leaf',
                                'learning_rate_values',
                                'learning_rate_index',
                                'polyLoss',
                                'polyLossEpsilon',
                                'focalLossGamma',
                                'loss']

        excluded_arguments = ['batch_size',
                              'epochs',
                              'restarts',
                              'restart_type',
                              'early_stopping_epochs',
                              'early_stopping_type',
                             ]

        for arg_key, arg_value in kwargs.items():
            if arg_key not in conditional_arguments and arg_key not in excluded_arguments:
                setattr(self, arg_key, arg_value)

        for conditional_argument in conditional_arguments:
            arg_key = conditional_argument
            if conditional_argument in kwargs.keys():
                arg_value = kwargs[arg_key]
            else:
                if arg_key != 'random_seed':
                    arg_value = self.__dict__[arg_key]
                    #setattr(self, arg_key, arg_value)
                else:
                    arg_value = self.__dict__['seed']
                    #setattr(self, 'seed', arg_value)

            if arg_key == 'random_seed':
                self.seed = arg_value
            elif arg_key == 'depth':
                self.depth = arg_value
            elif arg_key == 'initializer':
                self.initializer = arg_value

            elif arg_key == 'optimizer':
                self.optimizer = arg_value

            elif arg_key == 'learning_rate_leaf':
                self.learning_rate_leaf = arg_value

            elif arg_key == 'learning_rate_values':
                self.learning_rate_values = arg_value

            elif arg_key == 'learning_rate_index':
                self.learning_rate_index = arg_value

            elif arg_key == 'polyLoss':
                self.polyLoss = arg_value

            elif arg_key == 'polyLossEpsilon':
                self.polyLossEpsilon = arg_value

            elif arg_key == 'focalLossGamma':
                self.focalLossGamma = arg_value

            elif arg_key == 'loss':
                self.loss_name = arg_value


            self.internal_node_num_ = 2 ** self.depth - 1
            self.leaf_node_num_ = 2 ** self.depth

            leaf_classes_array_shape = (self.leaf_node_num_,) if self.number_of_classes == 2 or self.objective == 'regression' else(self.leaf_node_num_, self.number_of_classes)
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            tf.keras.utils.set_random_seed(int(self.seed))
            #tf.config.experimental.enable_op_determinism()

            if self.initializer == 'dataAware':
                self.split_values = tf.Variable(tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=self.seed)(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_values', dtype=tf.float32)

                self.split_index_array = tf.Variable(tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=self.seed)(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_index_array', dtype=tf.float32)

                self.leaf_classes_array = tf.Variable(tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=self.seed)(shape=leaf_classes_array_shape, dtype=tf.float32),trainable=True,name='leaf_classes_array', dtype=tf.float32)
            else:
                self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_values', dtype=tf.float32)

                self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables), dtype=tf.float32), trainable=True, name='split_index_array', dtype=tf.float32)

                self.leaf_classes_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.seed}})(shape=leaf_classes_array_shape, dtype=tf.float32),trainable=True,name='leaf_classes_array', dtype=tf.float32)

            if self.optimizer == 'QHAdam':
                self.optimizer_tree_split_index_array = QHAdamOptimizer(learning_rate=self.learning_rate_index)
                self.optimizer_tree_split_values = QHAdamOptimizer(learning_rate=self.learning_rate_values)
                self.optimizer_tree_leaf_classes_array = QHAdamOptimizer(learning_rate=self.learning_rate_leaf)
            elif self.optimizer == 'SWA':
                self.optimizer_tree_split_index_array = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_index), average_period=5)
                self.optimizer_tree_split_values = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_values), average_period=5)
                self.optimizer_tree_leaf_classes_array = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_leaf), average_period=5)   
            else:
                self.optimizer_tree_split_index_array = tf.keras.optimizers.get(self.optimizer)
                self.optimizer_tree_split_values = tf.keras.optimizers.get(self.optimizer)
                self.optimizer_tree_leaf_classes_array = tf.keras.optimizers.get(self.optimizer)

                self.optimizer_tree_split_index_array.learning_rate = self.learning_rate_index
                self.optimizer_tree_split_values.learning_rate = self.learning_rate_values
                self.optimizer_tree_leaf_classes_array.learning_rate = self.learning_rate_leaf

            if self.polyLoss:
                self.loss = poly1_cross_entropy(self.number_of_classes, epsilon=self.polyLossEpsilon, base_loss=self.loss_name, focalLossGamma=self.focalLossGamma)
            else:
                if self.polyLoss:
                    self.loss = poly1_cross_entropy(self.number_of_classes, epsilon=self.polyLossEpsilon, base_loss=loss, focalLossGamma=self.focalLossGamma)
                else:
                    if self.loss_name == 'crossentropy':
                        if self.number_of_classes == 2:
                            self.loss = tf.keras.losses.get('binary_crossentropy')
                        else:
                            self.loss = tf.keras.losses.get('categorical_crossentropy')
                    elif self.loss_name == 'binary_focal_crossentropy':
                        if self.number_of_classes == 2:
                            self.loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=self.focalLossGamma)
                        else:
                            self.loss = SparseCategoricalFocalLoss(gamma=self.focalLossGamma)
                    else:
                        self.loss = tf.keras.losses.get(self.loss_name)



    def adjust_preds_for_metric(self, preds, logits=False, denormalize=True):

        if logits:
            if self.objective == 'classification':
                if self.number_of_classes == 2:
                    preds = tf.sigmoid(preds)
                else:
                    if self.output_activation == "softmax":
                        preds = tf.keras.activations.softmax(preds)
                    elif self.output_activation == "entmax":
                        preds = entmax15(preds)
                    elif self.output_activation == "sparsemax":
                        preds = tfa.activations.sparsemax(preds)
            elif self.objective == 'regression':
                if denormalize:
                    preds = preds_batch = self.denormalize_labels(preds)


        if self.objective == 'classification':
            if self.number_of_classes == 2:
                preds_metric = tf.round(preds)
            else:
                preds_metric = tfa.seq2seq.hardmax(preds)
        if self.objective == 'regression':
            preds_metric = preds

        return preds_metric




    def normalize_labels(self,
                        labels: tf.Tensor):
        if self.normalize == 'mean':
            labels = (labels - self.data_mean) / self.data_std
        elif self.normalize == 'min-max':
            labels = (labels - self.data_min) / (self.data_max - self.data_min)

        return labels

    def denormalize_labels(self,
                           labels: tf.Tensor):
        if self.normalize == 'mean':
            labels = labels * self.data_std + self.data_mean
        elif self.normalize == 'min-max':
            labels = labels * (self.data_max - self.data_min) + self.data_min

        return labels

    def apply_dropout(self,
                      index_array: tf.Tensor,
                      training: bool):

        if self.dropout > 0.0 and training:
            row_index = tf.cast(tf.constant([i for i in range(index_array.shape[0])]), tf.int64)

            #tf.random.set_seed(self.seed)
            mask = tf.cast(tf.greater(np.random.uniform(0,1,(index_array.shape[0])), self.dropout), tf.float32)

            max_split = tf.stop_gradient(tf.argmax(index_array, axis=1))
            max_split_with_index = tf.transpose(tf.stack([row_index,max_split]))#tf.concat([row_index,max_split], axis=1)
            sparse_tensor = tf.SparseTensor(
                                  dense_shape=index_array.shape,#[index_array.shape[0], index_array.shape[1]],
                                  values=[0.0 for i in range(index_array.shape[0])],
                                  indices = max_split_with_index)
            dense_tensor = tf.sparse.to_dense(sparse_tensor, default_value = 1.0)

            dropout_mask = tf.cast(tf.greater(tf.math.add(dense_tensor, tf.expand_dims(mask, 1)), 0), tf.float32)

            index_array = index_array * dropout_mask

        else:
            index_array = index_array

        return index_array







def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """
    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True)
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs

        return outputs, grad_fn

    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param labels: reference answers vector int64[batch_size] \in [0, num_classes)
    :param logits: output matrix float32[batch_size, num_classes] (not actually logits :)
    :returns: elementwise loss, float32[batch_size]
    """
    assert logits.shape.ndims == 2 and labels.shape.ndims == 1
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param logits: "logits" matrix float32[batch_size, num_classes]
    :param labels: reference answers indicators, float32[batch_size, num_classes]
    :returns: elementwise loss, float32[batch_size]

    WARNING: this function does not propagate gradients through :labels:
    This behavior is the same as like softmax_crossentropy_with_logits v1
    It may become an issue if you do something like co-distillation
    """
    assert labels.shape.ndims == logits.shape.ndims == 2
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """ performs tf.nn.top_k over any chosen axis """
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """
    replicates the behavior of torch.gather for tf<=1.8;
    for newer versions use tf.gather with batch_dims
    :param values: tensor [d0, ..., dn]
    :param indices: int64 tensor of same shape as values except for gather_axis
    :param gather_axis: performs gather along this axis
    :returns: gathered values, same shape as values except for gather_axis
        If gather_axis == 2
        gathered_values[i, j, k, ...] = values[i, j, indices[i, j, k, ...], ...]
        see torch.gather for more detils
    """
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_support'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size



class DecisionTree:
    def __init__(self, split_thresholds, one_hot_encodings, class_probabilities):
        self.split_thresholds = split_thresholds
        self.one_hot_encodings = one_hot_encodings
        self.class_probabilities = class_probabilities
        self.max_depth = int(np.log2(len(class_probabilities)))
        self.root = self.build_tree()

    def build_tree(self, node_index=0):
        # If node index is greater than the number of internal nodes, we have reached a leaf node
        if node_index >= len(self.split_thresholds):
            return {'class': self.class_probabilities[node_index - len(self.split_thresholds)]}

        # Otherwise, create a new node with the corresponding threshold value and recursively build its children
        threshold_values = self.split_thresholds[node_index]
        one_hot_encoding = self.one_hot_encodings[node_index]
        node = {'threshold_values': threshold_values, 'one_hot_encoding': one_hot_encoding}
        node['left'] = self.build_tree(2 * node_index + 1)
        node['right'] = self.build_tree(2 * node_index + 2)

        return node

    def prune_tree(self, data, min_samples):
        if min_samples < 1: #int(min_samples) - min_samples != 0: #if float number
            min_samples = max(1, data.shape[0] * min_samples)
        
        data_complete = deepcopy(data)
        
        self._pass_node(self.root, data)
        
        self.root_unpruned = deepcopy(self.root)
        self._prune_node(self.root, data, data_complete, min_samples)

    def _pass_node(self, node, data):
        
        node['num_samples_passed'] = data.shape[0]
        #print(node)
        if 'class' in node:
            return

        left_indices = np.where(data[:, np.argmax(node['one_hot_encoding'])] <= node['threshold_values'][np.argmax(node['one_hot_encoding'])])[0]
        right_indices = np.where(data[:, np.argmax(node['one_hot_encoding'])] > node['threshold_values'][np.argmax(node['one_hot_encoding'])])[0]
                 
        #print('left_indices_test', np.where(data[:, np.argmax(node['one_hot_encoding'])] <= node['threshold_values'][np.argmax(node['one_hot_encoding'])]))
        #print('left_indices', left_indices)
        
            
        self._pass_node(node['left'], data[left_indices])
        self._pass_node(node['right'], data[right_indices])        
        
        
    def _prune_node(self, node, data, data_complete, min_samples):
        
        # If the node is a leaf, return
        if 'class' in node:
            return

        # Recursively prune the children of the node
        self._prune_node(node['left'], data, data_complete, min_samples)
        self._prune_node(node['right'], data, data_complete, min_samples)

        # Prune the node if the number of samples passing through it is less than the minimum
        
        samples_left = node['left']['num_samples_passed']
        samples_right = node['right']['num_samples_passed']
        
        #print('node', node)
        #print('node[left]', node['left'])
        
        if samples_left < min_samples and samples_right < min_samples:
            #print(node)
            if 'class' in node['left'] and 'class' in node['right']:
                #print(node['left']['class'], node['right']['class'])
                node['class'] = np.mean([node['left']['class'], node['right']['class']], axis=0)
                #print('node 1', node['class'])
                node['num_samples_passed'] = np.sum([node['left']['num_samples_passed'], node['right']['num_samples_passed']])
                node.pop('left', None)
                node.pop('right', None) 
            else:
                print('SHOULD NOT HAPPEN, CHECK PLEASE')
                return       

        else:
            if samples_left < min_samples:
                #print('node[left]', node['left'])
                #print('node[right]', node['right'])   
                if 'class' in node['right']:
                    node['class'] = node['right']['class']
                    #print('node 2', node['class'])
                    node['num_samples_passed'] = node['right']['num_samples_passed']
                    node.pop('left', None)
                    node.pop('right', None) 
                else:
                    new_node = deepcopy(node['right'])
                    node['left'] = new_node['left']
                    node['right'] = new_node['right']
                    node['one_hot_encoding'] = new_node['one_hot_encoding']
                    node['threshold_values'] = new_node['threshold_values']                
            elif samples_right < min_samples:
                #print('node[left]', node['left'])
                #print('node[right]', node['right'])
                if 'class' in node['left']:
                    node['class'] = node['left']['class']
                    #print('node 3', node['class'])
                    node['num_samples_passed'] = node['left']['num_samples_passed']
                    node.pop('left', None)
                    node.pop('right', None)
                else:
                    new_node = deepcopy(node['left'])
                    node['left'] = new_node['left']
                    node['right'] = new_node['right']
                    node['one_hot_encoding'] = new_node['one_hot_encoding']
                    node['threshold_values'] = new_node['threshold_values']
            else:
                return
        self._pass_node(self.root, data_complete)
                
    def predict(self, instance, node=None):
        # If no starting node is specified, start at the root of the tree
        if node is None:
            node = self.root

        # If we have reached a leaf node, return the corresponding class probability
        if 'class' in node:
            return node['class']

        # Otherwise, compare the instance's feature values to the node's threshold values and traverse the appropriate child
        threshold_values = node['threshold_values']
        one_hot_encoding = node['one_hot_encoding']
        feature_values = instance[one_hot_encoding == 1]
        if np.all(feature_values <= threshold_values):
            return self.predict(instance, node['left'])
        else:
            return self.predict(instance, node['right'])

    def evaluate(self, test_data, true_labels):
        num_correct = 0
        for i in range(len(test_data)):
            prediction = self.predict(test_data[i])
            if prediction == true_labels[i]:
                num_correct += 1
        accuracy = num_correct / len(test_data)
        return accuracy
    
    def extend_to_fully_grown(self):
        self.split_thresholds_unpruned = deepcopy(self.split_thresholds)
        self.one_hot_encodings_unpruned = deepcopy(self.one_hot_encodings)
        self.class_probabilities_unpruned = deepcopy(self.class_probabilities)
        self.root_pruned_extended = deepcopy(self.root)
        
        current_node_list = [self.root_pruned_extended]
        for current_depth in range(self.max_depth):
            current_node_list_new = []
            for current_node in current_node_list:
                if 'class' in current_node:
                    current_node_copy = deepcopy(current_node)
                    current_node.pop('class', None)
                    current_node['threshold_values'] = np.zeros_like(self.split_thresholds_unpruned[0])
                    current_node['one_hot_encoding'] = np.zeros_like(self.one_hot_encodings_unpruned[0])
                    current_node['left'] = current_node_copy
                    current_node['right'] = current_node_copy
                    
                current_node_list_new.append(current_node['left'])
                current_node_list_new.append(current_node['right'])
            current_node_list = current_node_list_new
            
        self.to_array_representation(root_type='pruned')
            

    def plot_tree_from_array(self, filename='./tree_tmp.png', plot_format='png'):
        dot = graphviz.Digraph()
        dot.node('0', 'Root')
        self._plot_subtree_from_array(dot, 0)
        dot.render(filename, format=plot_format, view=True)

    def _plot_subtree_from_array(self, dot, node_index):
        if node_index >= len(self.split_thresholds):
            node_label = f'Class: {self.class_probabilities[node_index - len(self.split_thresholds)]}'
        else:
            node_label = f'Feature {self.one_hot_encodings[node_index].argmax()}: <= {self.split_thresholds[node_index]}'
            left_child_index = 2 * node_index + 1
            right_child_index = 2 * node_index + 2
            dot.node(str(left_child_index), '')
            dot.node(str(right_child_index), '')
            dot.edge(str(node_index), str(left_child_index), 'True')
            dot.edge(str(node_index), str(right_child_index), 'False')
            self._plot_subtree_from_array(dot, left_child_index)
            self._plot_subtree_from_array(dot, right_child_index)
        dot.node(str(node_index), node_label)
        
    def plot_tree(self, filename='./tree_tmp', plot_format='png', root_type='current'): #initial, pruned_extended
        dot = graphviz.Digraph()
        if root_type == 'current':
            self._plot_subtree(dot, self.root)
        elif root_type == 'initial':
            self._plot_subtree(dot, self.root_unpruned)
        elif root_type == 'pruned_extended':
            self._plot_subtree(dot, self.root_pruned_extended)
        else:
            print('Root type ' + root_type + ' not existing, taking current root')
            self._plot_subtree(dot, self.root)
            
        dot.render(filename, format=plot_format, view=False)
        display(dot)
        #dot.render(filename, view=True)

    def _plot_subtree(self, dot, node):
        if 'class' in node:
            class_value = node["class"]
            num_samples_passed = node["num_samples_passed"]
            node_label = f'Class: {class_value:.3f} Num Samples: {num_samples_passed:.0f}'
        else:
            feature_index = node['one_hot_encoding'].argmax()
            threshold_value = node["threshold_values"][feature_index]
            num_samples_passed = node["num_samples_passed"]
            node_label = f'Feature {feature_index}: <= {threshold_value:.3f} Num Samples: {num_samples_passed:.0f}'
            left_child = node['left']
            right_child = node['right']
            dot.node(str(id(left_child)), '')
            dot.node(str(id(right_child)), '')
            dot.edge(str(id(node)), str(id(left_child)), 'True')
            dot.edge(str(id(node)), str(id(right_child)), 'False')
            self._plot_subtree(dot, left_child)
            self._plot_subtree(dot, right_child)
        dot.node(str(id(node)), node_label)
        
    def to_array_representation(self, root_type='pruned'): #, 'pruned'
        split_thresholds = []
        one_hot_encodings = []
        class_probabilities = []
        if root_type == 'initial':
            node_queue = [self.root_unpruned]
        elif root_type == 'pruned':
            try:
                node_queue = [self.root_pruned_extended]
            except:
                self.extend_to_fully_grown()
                node_queue = [self.root_pruned_extended]
        while node_queue:
            node = node_queue.pop(0)

            if 'class' in node:
                class_probabilities.append(node['class'])
            else:
                split_thresholds.append(node['threshold_values'])
                one_hot_encoding = np.zeros(len(node['threshold_values']), dtype=np.int)
                one_hot_encoding[node['one_hot_encoding'].argmax()] = 1
                one_hot_encodings.append(one_hot_encoding)

                node_queue.append(node['left'])
                node_queue.append(node['right'])

        self.split_thresholds = np.array(split_thresholds)
        self.one_hot_encodings = np.array(one_hot_encodings)
        self.class_probabilities = np.array(class_probabilities)

        return self.split_thresholds, self.one_hot_encodings, self.class_probabilities
    
    def count_nodes(self, node=None):
        if node is None:
            node = self.root
            if 'class' in node:
                leaf = 1
            else:
                internal = 1
        
        if 'class' in node:
            return 0, 1#1, 0

        left_internal, left_leaf = self.count_nodes(node['left'])
        right_internal, right_leaf = self.count_nodes(node['right'])
        internal = left_internal + right_internal
        leaf = left_leaf + right_leaf
        if 'left' in node or 'right' in node:
            internal += 1

        return internal, leaf