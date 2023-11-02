import tensorflow as tf
import tensorflow_addons as tfa

from functools import reduce
import numpy as np
from tqdm.notebook import tqdm
from collections.abc import Iterable
        
def make_batch_det(iterable, n=1):
    l = iterable.shape[0]
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]   
        
class DNDT(tf.Module):
    
    def __init__(
            self,
        
            num_features,
            num_classes,
            num_cut = None,
        
            temperature = 0.1,
            learning_rate = 0.1,
        
            random_seed = 42,
        
    ):

        self.num_features = num_features
        self.num_classes = num_classes
        
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        self.random_seed = random_seed
        tf.random.set_seed(self.random_seed)
        self.num_cut = num_cut
        
        if self.num_cut is None:
            self.num_cut_list = [1 for i in range(self.num_features)]
        else:
            self.num_cut_list = [num_cut for i in range(self.num_features)]
            
        self.num_leaf = np.prod(np.array(self.num_cut_list) + 1)  

        self.cut_points_list = [tf.Variable(tf.random.uniform([i])) for i in self.num_cut_list]
        self.leaf_score = tf.Variable(tf.random.uniform([self.num_leaf, self.num_classes]))

        self.opt = tf.keras.optimizers.get('adam')
        self.opt.learning_rate = self.learning_rate       
        
        
    def get_params(self):
        return {
            #'num_features': self.num_features,
            #'num_classes': self.num_classes,
            'num_cut': self.num_cut,
            
            'learning_rate': self.learning_rate,
            'temperature': self.temperature,
            
            'random_seed': self.random_seed,
        }      
        
        
    
    def set_params(self, **kwargs):
                
        conditional_arguments = ['random_seed',
                                'num_features',
                                'num_classes',
                                'num_cut',
                                'temperature',
                                'learning_rate']
        
        for conditional_argument in conditional_arguments:
            arg_key = conditional_argument
            if conditional_argument in kwargs.keys():
                arg_value = kwargs[arg_key]
            else:
                arg_value = self.__dict__[arg_key]
                #setattr(self, arg_key, arg_value)
                
            if arg_key == 'random_seed':
                self.random_seed = arg_value
            elif arg_key == 'num_features':
                self.num_features = arg_value
            elif arg_key == 'num_classes':
                self.num_classes = arg_value                
            elif arg_key == 'num_cut':
                self.num_cut = arg_value
            elif arg_key == 'temperature':
                self.temperature = arg_value                
            elif arg_key == 'learning_rate':
                self.learning_rate = arg_value                
                
        tf.random.set_seed(self.random_seed)
        if self.num_cut is None:
            self.num_cut_list = [1 for i in range(self.num_features)]

        self.num_leaf = np.prod(np.array(self.num_cut_list) + 1)  

        self.cut_points_list = [tf.Variable(tf.random.uniform([i])) for i in self.num_cut_list]
        self.leaf_score = tf.Variable(tf.random.uniform([self.num_leaf, self.num_classes]))

        self.opt = tf.keras.optimizers.get('adam')
        self.opt.learning_rate = self.learning_rate
        

    def fit(self, X, y, epochs=1000, batch_size=512, valid_data=None, early_stopping_epochs=25, verbosity=0):
    #def fit(self, X, y, epochs=1000, batch_size=10000, valid_data=None, early_stopping_epochs=1000, verbosity=0):
        
        X = tf.cast(X, tf.float32)
        
        y = tf.reshape(y,(-1,1)) if len(y.shape) == 1 else y           
        if y.shape[1] == 1:
            y = tf.cast(tf.keras.utils.to_categorical(y, num_classes=self.num_classes), tf.float32)
        else:
            y = tf.cast(y, tf.float32)
        
        if valid_data is not None:
            valid_data = (valid_data[0], tf.reshape(valid_data[1],(-1,1))) if len(valid_data[1].shape) == 1 else (valid_data[0], valid_data[1])    
            if valid_data[1].shape[1] == 1:
                valid_data = (tf.cast(valid_data[0], tf.float32), tf.cast(tf.keras.utils.to_categorical(valid_data[1], num_classes=self.num_classes), tf.float32))
            else:         
                valid_data = (tf.cast(valid_data[0], tf.float32), tf.reshape(tf.cast(valid_data[1], tf.float32),(-1,1)))
        
        complete_var_list = [self.leaf_score]

        for variable in self.cut_points_list:
            complete_var_list.append(variable)
        
        self.opt.build(complete_var_list)
        
        epochs_without_improvement = 0 
        minimum_loss_epoch = np.inf
        
        disable = True if verbosity == 0 else False
        for i in tqdm(range(epochs), disable=disable):
            loss_list = []
            y_pred_logits_list = []
            for X_batch, y_batch in zip(make_batch_det(X, batch_size), make_batch_det(y, batch_size)):
                loss_batch, y_pred_logits_batch = self.backward(X_batch, y_batch)
                loss_list.append(float(loss_batch))
                y_pred_logits_list.append(y_pred_logits_batch)

            loss = np.mean(loss_list)
            y_pred_logits = tf.concat(y_pred_logits_list, axis=0)             
            
            if valid_data is None:
                if loss < minimum_loss_epoch:
                    minimum_loss_epoch = loss
                    epochs_without_improvement = 0
                    
                    cut_points_list_stored = tf.identity(self.cut_points_list)
                    leaf_score_stored = tf.identity(self.leaf_score)
                    
                else:
                    epochs_without_improvement += 1
            else:
                y_pred_logits_valid = self.forward(valid_data[0])
                
                valid_loss = float(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_logits_valid, labels=valid_data[1])))
                
                if valid_loss < minimum_loss_epoch:
                    minimum_loss_epoch = valid_loss
                    epochs_without_improvement = 0
                    
                    cut_points_list_stored = tf.identity(self.cut_points_list)
                    leaf_score_stored = tf.identity(self.leaf_score)
                    
                else:
                    epochs_without_improvement += 1                

            if epochs_without_improvement >= early_stopping_epochs:
                try:                    
                    for value_index, value in enumerate(cut_points_list_stored):
                        self.cut_points_list[value_index].assign(value)
                    self.leaf_score.assign(leaf_score_stored)
                    
                except UnboundLocalError:
                    pass

                break 
                
            
            #if i % 200 == 0 and verbosity > 0:
            if verbosity > 0:
                print(loss)
        if verbosity > 0:
            print('error rate %.2f' % (1 - np.mean(np.argmax(self.forward(X), axis=1) == np.argmax(y, axis=1))))

    @tf.function(jit_compile=True)
    def forward(self, X, training=True):
        
        def tf_kron_prod(a, b):
            res = tf.einsum('ij,ik->ijk', a, b)
            res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
            return res


        def tf_bin(X, cut_points, training=True):
            # x is a N-by-1 matrix (column vector)
            # cut_points is a D-dim vector (D is the number of cut-points)
            # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
            D = cut_points.get_shape().as_list()[0]
            W = tf.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
            cut_points = tf.sort(cut_points)  # make sure cut_points is monotonically increasing
            b = tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points], 0))
            h = tf.matmul(X, W) + b
            
            if training:
                res = tf.nn.softmax(h / self.temperature)
            else:
                res = tfa.seq2seq.hardmax(h)            
            return res
        
        
        # cut_points_list contains the cut_points for each dimension of feature
        leaf = reduce(tf_kron_prod,
                      map(lambda z: tf_bin(X[:, z[0]:z[0] + 1], z[1], training=training), enumerate(self.cut_points_list)))
        return tf.matmul(leaf, self.leaf_score)
    
    def backward(self, X, y):
        
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
            tape1.watch([self.cut_points_list, self.leaf_score])

            y_pred_logits = self.forward(X)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_logits, labels=y))

        for var in self.cut_points_list:
            grads1 = tape1.gradient(loss, var)
            self.opt.apply_gradients(zip([grads1], [var]))           

        grads2 = tape1.gradient(loss, self.leaf_score)
        self.opt.apply_gradients(zip([grads2], [self.leaf_score]))    
        
        return loss, y_pred_logits
    
    def predict(self, X, batch_size=512, return_proba=False):
        X = tf.cast(X, tf.float32)
        
        y_pred_logits_list = []
        for X_batch in list(make_batch_det(X, batch_size)):        
            y_pred_logits_batch = self.forward(X_batch, training=False)
                    
            y_pred_logits_list.append(y_pred_logits_batch)
            
        y_pred_logits = tf.concat(y_pred_logits_list, axis=0)             
        y_pred = tf.keras.activations.softmax(y_pred_logits)
        
        if return_proba:
            if self.num_classes <= 2: 
                return np.array(y_pred[:,1])
            else:
                return np.array(y_pred)
            
        else:
            if self.num_classes <= 2: 
                return np.array(tf.round(y_pred[:,1]))
            else:
                return np.array(tf.argmax(y_pred, axis=1))