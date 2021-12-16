import tensorflow as tf
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
from keras import backend as K
from keras import objectives
from keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Flatten, RepeatVector, GRU, Convolution1D, Dropout,concatenate, BatchNormalization
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
import utilities
import numpy as np

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


class VAE():

    autoencoder = None
    
    def __init__(self):
        pass

    def load_grammar(self,grammar):
        self.grammar = grammar
        return

    
    def build(self,
              max_length = 277,
              dims = 2,
              n_props = 0,
              weights = None,
              freeze = False,
              lr = 0.005):

        if freeze:
            trainable = False
        else:
            trainable = True

        charset = self.grammar.grammar_string.split('\n')
        charset_length = len(charset)
        print(n_props)
        print('{}-D GVAE'.format(dims))

        # for obtaining mean and log variance of encoding distribution
        x = Input(shape=(max_length, charset_length))
        (z_mean, z_log_var) = self._encoderMeanVar(x, dims, max_length,trainable)
        self.encoderMV = Model(inputs=x, outputs=[z_mean, z_log_var])

        #For obtaining latent space encoding
    
        eps = utilities.sample_normal(K.shape(x)[0],dims)
        z_stat_1 = Input(shape= (dims,))
        z_stat_2 = Input(shape =(dims,))
        
        vae_loss, z = self._buildEncoder([z_stat_1,z_stat_2], eps,dims, max_length,charset_length,trainable)
        self.encoder = Model([z_stat_1,z_stat_2], z)

        #For decoding from latent space

        encoded_input = Input(shape=(dims,))
        x_decoded = self._buildDecoder(
            encoded_input,
            dims,
            max_length,
            charset_length,
            trainable
        )
        
        self.decoder = Model(encoded_input,x_decoded)

        #For obtaining property prediction data
        
        if n_props:
            print('Predicting {} properties'.format(n_props))
            p = Input(shape=(dims,))
            prop_vector = self._buildPredictor(p,n_props,trainable)
            self.predictor = Model(p,prop_vector)
        else:
            print('No property prediction selected')
        
        #Autoencoder with predictive capability
        inputs = []
        outputs = []
        loss = []
        metrics = []
        inputs.append(x)
#        ae_stats = concatenate([z_mean,z_log_var])
        _,intermediate_encoding = self._buildEncoder([z_mean,z_log_var],eps,dims,max_length,charset_length,trainable)
        decoding = self._buildDecoder(
            intermediate_encoding,
            dims,
            max_length,
            charset_length,
            trainable
            )
        if not freeze:
            outputs.append(decoding)
            loss.append(vae_loss)
            metrics.append('accuracy')
        if n_props:
            ae_property_vector = self._buildPredictor(z_mean,n_props,trainable)
            for i in range(n_props):
                outputs.append(ae_property_vector[i])
                loss.append('mse')
            metrics.append('mae')
            
        self.autoencoder = Model(inputs=inputs,outputs=outputs)

        self.encoderMV.summary()
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()

        if weights:
            self.encoderMV.load_weights(weights, by_name = True)
            self.encoder.load_weights(weights, by_name = True)
            self.decoder.load_weights(weights, by_name = True)
            self.autoencoder.load_weights(weights,by_name=True)
            if n_props:
                self.predictor.load_weights(weights,by_name = True)


        #Learning rate block
        if n_props:
            optimizer = optimizers.Adam(lr=lr)
            print('Learning rate is: {}'.format(lr))
        else:
            optimizer = optimizers.Adam(lr=lr)
            print('Learning rate is: {}'.format(lr))




        print('Using {} optimizer'.format(optimizer))

        alpha = K.variable(50)
        
        def sigmoid(var):
             return 50*np.exp(-0.1*var+0.1*50)/(1+np.exp(-0.1*var+0.1*50))

        loss_list = []
        loss_list.append(alpha)
        for i in range(n_props):
            loss_list.append(1)

        self.autoencoder.compile(optimizer = optimizer,
                                 loss = loss,
                                 metrics = metrics,
                                 loss_weights=loss_list)
        
        return

    def _encoderMeanVar(self, x, dims, max_length, trainable, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1', trainable = trainable)(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2', trainable = trainable)(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3', trainable = trainable)(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1', trainable = trainable)(h)

        z_mean = Dense(dims, name='z_mean', activation = 'linear', trainable = trainable)(h)
        z_log_var = Dense(dims, name='z_log_var', activation = 'linear', trainable = trainable)(h)
        return z_mean, z_log_var


    def _buildEncoder(self, z_stat, epsilon, dims, max_length, charset_length, trainable):

        def get_slice(inp):
            o1 = inp[0]
            o2 = inp[1]
            return [o1,o2]

        z_mean,z_log_var = Lambda(get_slice)(z_stat)

        if trainable:
            z_mean,z_log_var = utilities.KLDLossLayer()([z_mean,z_log_var]) #Dummy layer that adds KLDivergence term to loss

        #Convert mask and index array into Keras tensor
            
        K_masks = K.variable(self.grammar.masks) 
        K_index_array =K.variable(self.grammar.index_of_index) 

        sampler = Lambda(utilities.sampling)([z_mean,z_log_var])
        vae_loss = utilities.masked_xent_loss(K_masks,K_index_array,max_length,charset_length)
        return (vae_loss, sampler)

    def _buildDecoder(self, z, dims, max_length, charset_length, trainable):
        h = Dense(dims, name='latent_input', activation = 'relu', trainable = trainable)(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1', trainable = trainable)(h)
        h = GRU(501, return_sequences = True, name='gru_2', trainable = trainable)(h)
        h = GRU(501, return_sequences = True, name='gru_3', trainable = trainable)(h)
        output= TimeDistributed(Dense(charset_length, trainable = trainable), name='decoded_mean')(h) # don't do softmax, we do this in the loss now
        return output

    def _buildPredictor(self,z_mean,n_props, trainable):

       
        trainable=True
        outputs = []
        if trainable:
                
            


            print('Prediction based on entire vector')
            for i in range(n_props):
                pred_out = Dense(1,activation='linear',name='predictor_out_{}'.format(i))(z_mean)
                outputs.append(pred_out)
            
 
    


              return outputs 
    


        

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, weights, dims = 2, max_length=277,n_props = 0,freeze=False,lr=0.005):
        self.build(max_length = max_length, weights = weights, dims=dims,n_props=n_props,freeze=freeze,lr=lr)


    def encode(self,one_hot):
        """
        Encode one-hot representation of smiles
        into latent space. Returns mean of encoding.
        """
        mid = np.asarray(self.encoderMV.predict(one_hot))
        return self.encoder.predict([mid[0],mid[1]])



#        return self.encoderMV.predict(one_hot)[0]
        
    def encode_MV(self,one_hot):
        return self.encoderMV.predict(one_hot)[0]


    
    def decode(self,z):
               """
        # assert z.ndim ==2  Why? Number of dimensions shouldnt matter

        unmasked = self.decoder.predict(z)

        X_hat = utilities.masked_sample(self.grammar,unmasked)
        
        #Convert from one-hot to production rules

        prod_seq = [self.grammar.grammar.productions()[X_hat[t].argmax()] for t in range(X_hat.shape[0])]

        decoding = utilities.prods_to_decoding(prod_seq)

        return decoding
    
    def prop_predict(self,one_hots):
        enc = self.encoderMV.predict(one_hots)[0]
        output = self.predictor.predict(enc)
        return output
