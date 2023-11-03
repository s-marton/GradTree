import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn
from copy import deepcopy
import category_encoders as ce
import pandas as pd
import math
from focal_loss import SparseCategoricalFocalLoss
     
class GradTree(tf.keras.Model):
    def __init__(self, 
                 params, 
                 args):

        params.update(args)
        self.config = None

        super(GradTree, self).__init__()      
        self.set_params(**params)


    @tf.function(jit_compile=True)
    def call(self, inputs, training):
        output = self.output_layer(inputs)

        if self.objective == 'regression' or self.objective == 'binary':                                   
            result = tf.einsum('be->b', output)
        else:                    
            result = tf.einsum('bec->bc', output)

        if self.objective == 'regression' or self.objective == 'binary':   
            result = tf.expand_dims(result, 1)

        return result

    

    def fit(self, 
            X_train, 
            y_train, 
            X_val=None,
            y_val=None,
            **kwargs):

        low_cardinality_indices = []
        high_cardinality_indices = []
        num_columns = []

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)
        
        for column, column_index in enumerate(X_train.columns):
            if column_index in self.cat_idx:
                if len(X_train.iloc[:,column_index].unique()) < 10:
                    low_cardinality_indices.append(column)
                else:
                    high_cardinality_indices.append(column)
            else:
                num_columns.append(column)
        
        self.encoder_loo = ce.LeaveOneOutEncoder(cols=high_cardinality_indices)
        self.encoder_loo.fit(X_train, y_train)
        X_train = self.encoder_loo.transform(X_train)
        X_val = self.encoder_loo.transform(X_val)
        
        self.encoder_ohe = ce.OneHotEncoder(cols=low_cardinality_indices)
        self.encoder_ohe.fit(X_train)
        X_train = self.encoder_ohe.transform(X_train)
        X_val = self.encoder_ohe.transform(X_val)
        
        self.median_train = X_train.median(axis=0)
        X_train = X_train.fillna(self.median_train)
        X_val = X_val.fillna(self.median_train)

        self.cat_columns_preprocessed = []
        for column, column_index in enumerate(X_train.columns):
            if column not in num_columns:
                self.cat_columns_preprocessed.append(column_index)
        quantile_noise = 1e-4
        quantile_train = np.copy(X_train.values).astype(np.float64)
        np.random.seed(42)
        stds = np.std(quantile_train, axis=0, keepdims=True)
        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
        quantile_train += noise_std * np.random.randn(*quantile_train.shape)    

        quantile_train = pd.DataFrame(quantile_train, columns=X_train.columns, index=X_train.index)

        self.normalizer = sklearn.preprocessing.QuantileTransformer(
                                                                    n_quantiles=min(quantile_train.shape[0], 1000),
                                                                    output_distribution='normal',
                                                                    )

        self.normalizer.fit(quantile_train.values.astype(np.float64))
        X_train = self.normalizer.transform(X_train.values.astype(np.float64))
        X_val = self.normalizer.transform(X_val.values.astype(np.float64))

        self.mean = np.mean(y_train)
        self.std = np.std(y_train)

        self.number_of_variables = X_train.shape[1]
        if self.objective == 'classification' or self.objective == 'binary':
            self.number_of_classes = len(np.unique(y_train))
            self.class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)

            self.class_weight_dict = {}
            for i in range(self.number_of_classes):
                self.class_weight_dict[i] = self.class_weights[i]

        else:
            self.number_of_classes = 1
            self.class_weights = np.ones_like(np.unique(y_train))
            self.class_weight_dict = None

        self.build_model()

        self.compile(loss=self.loss_name, metrics=self.metrics_name, mean=self.mean, std=self.std, class_weight=self.class_weights)

        train_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(X_train), tf.float32),
                                                         tf.dtypes.cast(tf.convert_to_tensor(y_train), tf.float32)))

        train_data = (train_data
                .shuffle(32_768)
                .cache()
                .batch(batch_size=self.batch_size, drop_remainder=False) 
                .prefetch(tf.data.AUTOTUNE)      
                    )

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_data = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(tf.convert_to_tensor(validation_data[0]), tf.float32), 
                                                             tf.dtypes.cast(tf.convert_to_tensor(validation_data[1]), tf.float32)))


            validation_data = (validation_data
                    .cache()
                    .batch(batch_size=self.batch_size, drop_remainder=False) 
                    .prefetch(tf.data.AUTOTUNE)      
                         )   

            monitor = 'val_loss'    
        else:
            monitor = 'loss'


        if 'callbacks' not in kwargs.keys():
            callbacks = []

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                          patience=self.early_stopping_epochs, 
                                                          min_delta=1e-3,
                                                          restore_best_weights=True)
        callbacks.append(early_stopping)

        if 'reduce_lr' in kwargs.keys() and kwargs['reduce_lr']:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=self.early_stopping_epochs//3)
            callbacks.append(reduce_lr)

        super(GradTree, self).fit(train_data,
                                validation_data = validation_data,
                                epochs = self.epochs,
                                callbacks = callbacks,
                                class_weight = self.class_weight_dict,
                                verbose=self.verbose,
                                **kwargs)
          
    def build_model(self):
        self.config['mean'] = self.mean
        self.config['std'] = self.std
        self.config['number_of_classes'] = self.number_of_classes
        self.config['number_of_variables'] = self.number_of_variables

        self.output_layer = GradTreeBlock(**self.config) 


    def compile(self, 
        loss, 
        metrics, 
        **kwargs):

        if self.objective == 'classification':

            if loss == 'crossentropy':
                if not self.focal_loss:
                    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits) #tf.keras.losses.get('categorical_crossentropy')
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=2, class_weight=self.class_weights, from_logits=from_logits)
            else:
                loss_function = tf.keras.losses.get(loss)  
                try:
                     loss_function.from_logits = self.from_logits
                except:
                    pass        
        elif self.objective == 'binary':
            if loss == 'crossentropy':
                if not self.focal_loss:
                    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=self.from_logits) #tf.keras.losses.get('binary_crossentropy')
                else:
                    loss_function = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=2, apply_class_balancing=self.apply_class_balancing, from_logits=self.from_logits)
            else:
                loss_function = tf.keras.losses.get(loss)  
                try:
                     loss_function.from_logits = self.from_logits
                except:
                    pass   

        elif self.objective == 'regression':
            loss_function = loss_function_regression(loss_name=loss, mean=kwargs['mean'], std=kwargs['std'])

        loss_function = loss_function_weighting(loss_function, temp=self.temperature)
        self.index_optimizer = get_optimizer_by_name(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate_index, warmup_steps=0, cosine_decay_steps=self.cosine_decay_steps)
        self.values_optimizer = get_optimizer_by_name(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate_values, warmup_steps=0, cosine_decay_steps=self.cosine_decay_steps)
        self.leaf_optimizer = get_optimizer_by_name(optimizer_name=self.optimizer_name, learning_rate=self.learning_rate_leaf, warmup_steps=0, cosine_decay_steps=self.cosine_decay_steps)

        if metrics is None:
            metrics = []
        elif isinstance(metrics, list):
            metrics_new = []
            for name in metrics:
                if name == 'F1':
                    metrics.append(F1ScoreSparse(average='macro', num_classes=self.number_of_classes, threshold=0.5))
                elif name == 'Accuracy':
                    metrics.append(AccuracySparse(average='macro', num_classes=self.number_of_classes, threshold=0.5))
                elif name == 'R2':
                    metrics.append(R2ScoreTransform(mean=kwargs['mean'], std=kwargs['std']))
            metrics = metrics_new
        else:
            if metrics == 'F1':
                metrics = [F1ScoreSparse(average='macro', num_classes=self.number_of_classes, threshold=0.5)]
            elif metrics == 'Accuracy':
                metrics = [AccuracySparse(average='macro', num_classes=self.number_of_classes, threshold=0.5)]
            elif metrics == 'R2':
                metrics = [R2ScoreTransform(mean=kwargs['mean'], std=kwargs['std'])]

        super(GradTree, self).compile(loss=loss_function, metrics=metrics)
        

    def train_step(self, data):
    
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        if not self.built:
            self(x) 

        with tf.GradientTape() as index_tape:
            with tf.GradientTape() as values_tape:
                with tf.GradientTape() as leaf_tape:                        
                    index_tape.watch(self.output_layer.split_index_array)
                    values_tape.watch(self.output_layer.split_values)
                    leaf_tape.watch(self.output_layer.leaf_classes_array)
                    
                    y_pred = self(x, training=True) 

                    # Compute the loss
                    loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        if self.split_index_trainable:
            index_gradients = index_tape.gradient(loss, [self.output_layer.split_index_array])
            self.index_optimizer.apply_gradients(zip(index_gradients, [self.output_layer.split_index_array]))
             
        if self.split_values_trainable:
            values_gradients = values_tape.gradient(loss, [self.output_layer.split_values])
            self.values_optimizer.apply_gradients(zip(values_gradients, [self.output_layer.split_values]))
            
        if self.leaf_trainable:
            leaf_gradients = leaf_tape.gradient(loss, [self.output_layer.leaf_classes_array])  
            self.leaf_optimizer.apply_gradients(zip(leaf_gradients, [self.output_layer.leaf_classes_array]))        

        # Update metrics (optional)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}
 

    def predict(self, X, batch_size=64, verbose=0):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self.encoder_loo.transform(X)
        X = self.encoder_ohe.transform(X)
        X = X.fillna(self.median_train)

        X = self.normalizer.transform(X.values.astype(np.float64))

        preds = super(GradTree, self).predict(X, batch_size, verbose=verbose)
        preds = tf.convert_to_tensor(preds)
        if self.objective == 'regression':
            preds = preds * self.std + self.mean  
        else:
            if self.from_logits:
                if self.objective == 'binary':
                    preds = tf.math.sigmoid(preds)
                elif self.objective == 'classification':
                    preds = tf.keras.activations.softmax(preds) 

        if self.objective == 'binary':
            preds = tf.stack([1-tf.squeeze(preds), tf.squeeze(preds)], axis=-1)      

        return preds.numpy()

    def set_params(self, **kwargs): 
                
        if self.config is None:
            self.config = {
                'objective': 'binary',

                'depth': 6,
                'n_estimators': 1,

                'learning_rate_index': 0.01,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.01,
                'temperature': 0.0,

                'optimizer': 'SWA',
                'cosine_decay_steps': 0,

                'initializer': 'RandomNormal',

                'from_logits': True,
                'apply_class_balancing': True,

                'metrics': ['F1'], # F1, Accuracy, R2
                'random_seed': 42,
                'verbose': 0,
            }


        self.config.update(kwargs)

        if 'loss' not in self.config.keys():
            if self.config['objective'] == 'classification' or self.config['objective'] == 'binary':
                self.config['loss'] = 'crossentropy'
                self.config['focal_loss'] = False
            elif self.config['objective'] == 'regression':
                self.config['loss'] = 'mse'
                self.config['focal_loss'] = False

        self.config['optimizer_name'] = self.config.pop('optimizer')
        self.config['loss_name'] = self.config.pop('loss')
        if 'metrics' in self.config.keys():
            self.config['metrics_name'] = self.config.pop('metrics')
        else: 
            self.config['metrics_name'] = []
        for arg_key, arg_value in self.config.items():
            setattr(self, arg_key, arg_value)     

        self.leaf_trainable = True
        self.split_index_trainable = True
        self.split_values_trainable = True
        self.weights_trainable = True
        
        tf.keras.utils.set_random_seed(self.random_seed)   
                 
                
    def get_params(self):
        return self.config    

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            'depth': trial.suggest_int("depth", 3, 10),

            'learning_rate_index': trial.suggest_float("learning_rate_index", 0.0001, 0.25),
            'learning_rate_values': trial.suggest_float("learning_rate_values", 0.0001, 0.25),
            'learning_rate_leaf': trial.suggest_float("learning_rate_leaf", 0.0001, 0.25),

            'cosine_decay_steps': trial.suggest_categorical("cosine_decay_steps", [0, 100, 1000]),
        }

        try:
            if args['objective'] != 'regression':
                params['focal_loss'] = trial.suggest_categorical("focal_loss", [True, False])
                params['temperature'] = trial.suggest_categorical("temperature", [0, 0.25])
        except:
            if self.objective  != 'regression':
                params['focal_loss'] = trial.suggest_categorical("focal_loss", [True, False])
                params['temperature'] = trial.suggest_categorical("temperature", [0, 0.25])
        return params

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            'depth': rs.randint(3, 10),

            'learning_rate_index': rs.uniform(0.0001, 0.25),
            'learning_rate_values': rs.uniform(0.0001, 0.25),
            'learning_rate_leaf': rs.uniform(0.0001, 0.25),

            'cosine_decay_steps': rs.choice([0, 100, 1000], p=[0.5, 0.25, 0.25]),
        }

        if self.objective != 'regression':
            params['focal_loss'] = rs.choice([True, False])
            params['temperature'] = rs.choice([1, 1/3, 1/5, 1/7, 1/9, 0], p=[0.1, 0.1, 0.1, 0.1, 0.1,0.5]),

        return params

    @classmethod
    def default_parameters(cls):
        params = {
            'depth': 6,
            'n_estimators': 1,

            'learning_rate_index': 0.01,
            'learning_rate_values': 0.01,
            'learning_rate_leaf': 0.01,

            'optimizer': 'SWA',
            'cosine_decay_steps': 0,
            'temperature': 0.0,

            'initializer': 'RandomNormal',

            'loss': 'crossentropy',
            'focal_loss': False,

            'from_logits': True,
            'apply_class_balancing': True,
        }        

        return params

    
    
class GradTreeBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 name='GradTreeBlock',
                 trainable=True,
                 dtype='float',
                 **kwargs):
        super(GradTreeBlock, self).__init__()

        for arg_key, arg_value in kwargs.items():
            setattr(self, arg_key, arg_value)  
                
        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        tf.keras.utils.set_random_seed(self.random_seed)
  
        
    def build(self, input_shape):

        tf.keras.utils.set_random_seed(self.random_seed)

        self.data_shape = self.number_of_variables
        self.number_of_variables = input_shape[-1]   
        self.selected_variables = self.number_of_variables

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

        leaf_classes_array_shape = [self.n_estimators,self.leaf_node_num_,] if self.objective == 'binary' or self.objective == 'regression' else [self.n_estimators, self.leaf_node_num_, self.number_of_classes]
                    
        self.split_values = self.add_weight(name="split_values",  
                                            initializer=tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + 2}}),
                                            trainable=True,
                                            shape=[self.n_estimators,self.internal_node_num_, self.selected_variables])
        self.split_index_array = self.add_weight(name="split_index_array",
                                                 initializer=tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + 3}}),
                                                 trainable=True,
                                                 shape=[self.n_estimators,self.internal_node_num_, self.selected_variables])
        self.leaf_classes_array = self.add_weight(name="leaf_classes_array",  
                                                 initializer=tf.keras.initializers.get({'class_name': self.initializer, 'config': {'seed': self.random_seed + 4}}),
                                                 trainable=True,
                                                 shape=leaf_classes_array_shape)      


    def call(self, inputs, training, output_weights=False):

        # einsum syntax: e - number of estimators; b - batch size; d - depth, n - number of features, i - internal node num by estimator, l - leaf node num by estimator, c - number of classes
        X_estimator = tf.expand_dims(inputs, 1)

        #entmax transformaton
        split_index_array = entmax15(self.split_index_array)
  
        #use ST-Operator to get one-hot encoded vector for feature index
        adjust_constant = tf.stop_gradient(split_index_array - tfa.seq2seq.hardmax(split_index_array))
        split_index_array = split_index_array - adjust_constant        

        #generate tensor for further calculation: 
        # - internal_node_index_list holds the indices for the internal nodes traversed for each path (there are l paths) in the tree
        # - for each estimator and for each path in each estimator, the tensors hold the information for all internal nodex traversed
        # - the resulting shape of the tensors is (e, l, d, n):
        #       - e is the number of estimators
        #       - l the number of leaf nodes  (i.e. the number of paths)
        #       - d is the depth (i.e. the length of each path)
        #       - n is the number of variables (one value is stored for each variable)
        split_index_array_selected = tf.gather(split_index_array, self.internal_node_index_list, axis=1)
        split_values_selected = tf.gather(self.split_values, self.internal_node_index_list, axis=1)

        # as split_index_array_selected is one-hot-encoded, taking the sum over the last axis after multiplication results in selecting the desired value at the index
        s1_sum = tf.einsum("eldn,eldn->eld", split_values_selected, split_index_array_selected)
        s2_sum = tf.einsum("ben,eldn->beld", X_estimator, split_index_array_selected)

        # calculate the split (output shape: (b, e, l, d))
        node_result = (tf.nn.softsign(s1_sum-s2_sum) + 1) / 2
        #use round operation with ST operator to get hard decision for each node
        node_result_corrected = node_result - tf.stop_gradient(node_result - tf.round(node_result))

        #reduce the path via multiplication to get result for each path (in each estimator) based on the results of the corresponding internal nodes (output shape: (b, e, l))
        p = tf.reduce_prod(((1-self.path_identifier_list)*node_result_corrected + self.path_identifier_list*(1-node_result_corrected)), axis=3)
 

        #get raw prediction for each estimator
        #optionally transform to probability distribution before weighting
        if self.objective == 'regression':
            layer_output = tf.einsum('el,bel->be', self.leaf_classes_array, p)                
        elif self.objective == 'binary':
            if self.from_logits:
                layer_output = tf.einsum('el,bel->be', self.leaf_classes_array, p)               
            else:
                layer_output = tf.math.sigmoid(tf.einsum('el,bel->be', self.leaf_classes_array, p))
        elif self.objective == 'classification':
            if self.from_logits:
                layer_output = tf.einsum('elc,bel->bec', self.leaf_classes_array, p)
            else:
                layer_output = tf.keras.activations.softmax(tf.einsum('elc,bel->bec', self.leaf_classes_array, p))

        return layer_output
      
    @classmethod
    def from_config(cls, config):
        return cls(**config)    
    
         
def entmax15(inputs, axis=-1):

    # Implementation taken from: https://github.com/deep-spin/entmax/tree/master/entmax 

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


class R2ScoreTransform(tf.keras.metrics.Metric):
    def __init__(self, mean=0, std=1, name='r2score_transform', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std
        self.metric = tfa.metrics.RSquare()

    def update_state(self, y_true, y_pred, sample_weight=None):

        if not tf.keras.backend.learning_phase():
            y_true = (y_true - self.mean) / self.std
            
        # Update precision and recall
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        r2_score = self.metric.result()
        return r2_score

    def reset_states(self):
        self.metric.reset_states()        
        

def loss_function_weighting(loss_function, temp=0.25): 

    # Implementation of "Stochastic Re-weighted Gradient Descent via Distributionally Robust Optimization" from https://arxiv.org/abs/2306.09222

    loss_function.reduction = tf.keras.losses.Reduction.NONE
    def _loss_function_weighting(y_true, y_pred):
        loss = loss_function(y_true, y_pred)

        if temp > 0:
            clamped_loss = tf.clip_by_value(loss, clip_value_min=float('-inf'), clip_value_max=temp)

            out = loss * tf.stop_gradient(tf.exp(clamped_loss / (temp + 1)))
        else:
            out = loss
        
        return tf.reduce_mean(out)
    return _loss_function_weighting

def loss_function_regression(loss_name, mean, std): #mean, log, 
    loss_function = tf.keras.losses.get(loss_name)                                   
    def _loss_function_regression(y_true, y_pred):
        #if tf.keras.backend.learning_phase():
        y_true = (y_true - mean) / std

        loss = loss_function(y_true, y_pred)
        
        return loss
    return _loss_function_regression

def _threshold_and_support(input, dim=-1):
    Xsrt = tf.sort(input, axis=dim, direction='DESCENDING')

    rho = tf.range(1, tf.shape(input)[dim] + 1, dtype=input.dtype)
    mean = tf.math.cumsum(Xsrt, axis=dim) / rho
    mean_sq = tf.math.cumsum(tf.square(Xsrt), axis=dim) / rho
    ss = rho * (mean_sq - tf.square(mean))
    delta = (1 - ss) / rho

    delta_nz = tf.maximum(delta, 0)
    tau = mean - tf.sqrt(delta_nz)

    support_size = tf.reduce_sum(tf.cast(tau <= Xsrt, tf.int32), axis=dim)
    tau_star = tf.gather(tau, support_size - 1, batch_dims=-1)
    return tau_star, support_size

def get_optimizer_by_name(optimizer_name, learning_rate, warmup_steps, cosine_decay_steps):

    if cosine_decay_steps > 0:
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                                                            initial_learning_rate=learning_rate,
                                                                            first_decay_steps=cosine_decay_steps,
                                                                            #first_decay_steps=steps_per_epoch,
                                                                        )

    if optimizer_name== 'SWA':
        optimizer = tfa.optimizers.SWA(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), average_period=5)
    elif optimizer_name== 'AdamW':
        optimizer = tf.keras.optimizers.AdamW()  
    else:
        optimizer = tf.keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = learning_rate
                
    return optimizer



class F1ScoreSparse(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='macro', threshold=0.5, name='f1score_sparse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.metric = tf.keras.metrics.F1Score(average=average, threshold=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse labels to dense format
        if self.num_classes > 2:
            y_true_dense = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=self.num_classes)
        else:
            y_true_dense = y_true
        # Update precision and recall
        self.metric.update_state(y_true_dense, y_pred, sample_weight)

    def result(self):
        f1_score = self.metric.result()
        return f1_score

    def reset_states(self):
        self.metric.reset_states()
        
    def get_config(self):
        config = super(F1ScoreSparse, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'metric': self.metric,
        })
        
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)       


class AccuracySparse(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='macro', threshold=0.5, name='accuracy_sparse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.metric = tf.keras.metrics.Accuracy(average=average, threshold=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse labels to dense format
        if self.num_classes > 2:
            y_true_dense = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=self.num_classes)
        else:
            y_true_dense = y_true
        # Update precision and recall
        self.metric.update_state(y_true_dense, y_pred, sample_weight)

    def result(self):
        f1_score = self.metric.result()
        return f1_score

    def reset_states(self):
        self.metric.reset_states()
        
    def get_config(self):
        config = super(AccuracySparse, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'metric': self.metric,
        })
        
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)       


