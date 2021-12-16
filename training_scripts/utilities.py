"""
Various quality of life functions for autoencoder and
Grammar scripts
"""

import nltk
from keras import backend as K
from keras.layers import Layer,Lambda
from keras import objectives
import tensorflow as tf
import numpy as np

def get_tokenizer(grammar):
    """
    Pass grammar object to obtain a tokenizer
    that will split smiles string correcty
    even with character length greater than 2.
    Note: Will need to include some updated
    smiles parser to replace ring notations
    geater than 11 with some unique character.
    Will also want to update this script so that
    regular smiles can be tokenized as well
    """
    if isinstance(grammar,list): #For character models
        long_tokens = list(filter(lambda c: len(c)>1,grammar)) #Filter out characters w/ length greater than 1
    else:
        long_tokens = list(filter(lambda c: len(c)>1,grammar._lexical_index.keys())) #For grammar models
    #Create list of characters not utilized in smiles grammar
    replacement_list = ['!',
                        '$',
                        '%',
                        '^',
                        '&',
                        '<',
                        '>'
                        ]
    replacements = [] #Initialize list of replacement characters
    for i in range(len(long_tokens)): #Assign a unique replacement to each long character
        replacements.append(replacement_list[i])

    def tokenize(smiles):
        """
        Replaces long characters in smiles
        with unique token, splits the string,
        and then reinserts correct character
        """
        for i,e in enumerate(long_tokens): #Replace each instance of long character with corresponding token
            smiles = smiles.replace(e,replacements[i])
        tokens = []

        for s in smiles: #Iterate over characters in smiles string
            try:
                i = replacements.index(s) #If character is a dummy token, get index within replacement list
                tokens.append(long_tokens[i]) #Replace dummy token with correct sequence
            except ValueError:
                tokens.append(s) #If character is not a dummy token, add to output sequence
        return tokens

    return tokenize #Function gives tokenize function as output


def pop_or_nothing(stack):
   
    try:
        return stack.pop()
    except IndexError:
        return 'Nothing'

def sample_normal(batch_size,dims,mean=0,stddev = 0.01):
    epsilon = K.random_normal(shape=(batch_size, dims), mean=0., stddev = stddev)
    return epsilon

def sampling(args):
    z_mean_,z_log_var_ = args
    batch_size=K.shape(z_mean_)[0]
    dims = K.shape(z_mean_)[1]
    epsilon = sample_normal(batch_size,dims)
    sample = z_mean_ +K.exp(z_log_var_ /2.0)*epsilon
    return sample


class KLDLossLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDLossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        ###For pretraining, remove the factor of 750
        self.add_loss(750.0*K.mean(kl_batch), inputs=inputs)

        return inputs

def slicer(array,indicies):
    """
    Allows for indexing of keras objects
    """
    outputs = []
    for i in indicies:
        outputs.append(tf.gather_nd(array,[indicies[i]]))

    return outputs


def masked_xent_loss(masks ,index_array,max_length,charset_length):

    def conditional(x_true,x_pred):
        most_likely = K.argmax(x_true)
        most_likely = tf.reshape(most_likely,[-1]) # flatten most_likely
        ix2 = tf.expand_dims(tf.gather(index_array, most_likely),1) # index ind_of_ind with res
        ix2 = tf.cast(ix2, tf.int32) # cast indices as ints 
        M2 = tf.gather_nd(masks, ix2) # get slices of masks_K with indices
        M3 = tf.reshape(M2, [-1,max_length,charset_length]) # reshape them
        P2 = tf.multiply(K.exp(x_pred),M3) # apply them to the exp-predictions
        normalized_prediction = tf.divide(P2,K.sum(P2,axis=-1,keepdims=True)) 
        return normalized_prediction
    def vae_loss(x_true,x_pred):
        x_pred = conditional(x_true,x_pred)
        x_true = K.flatten(x_true)
        x_pred = K.flatten(x_pred)
        xent = objectives.binary_crossentropy(x_true,x_pred)
        xent *= max_length
        return xent
    return vae_loss
    

def masked_sample(grammar,unmasked):
    """
    Samples from one-hot logit vector
    using masked prediction algorithm
    """
    unmasked = np.squeeze(unmasked)
    X_hat = np.zeros_like(unmasked)
    
    #Initialize stack and add smiles as first entry
    
    S = []
    S.append(str(grammar.start_index))
#    S = np.asarray(S,dtype=object)
    
    #Loop over timesteps (max_length of rules).
    #For each timestep, pop first term from S(tack)
    #Term will give masks to apply to logit matrix
    #Sample from logit matrix, add resulting production 
    #rules to stack and continue until max timestep is
    #Reached

    eps = 1e-100 #Not sure why this is in orginal code, might be to prevent x/0

    # for t in range(unmasked.shape[1]):
    #     next_nonterminal = [grammar.lhs_map[pop_or_nothing(a)] for a in S]
    #     mask = grammar.masks[next_nonterminal]
    #     masked_logit = np.exp(unmasked[:,t,:])*mask + eps
    #     gumbel = np.random.gumbel(size=masked_logit.shape)
    #     sampled_output = np.argmax(gumbel+np.log(masked_logit),axis=-1)
    #     X_hat[np.arange(unmasked.shape[0]),t,sampled_output] =1.0
    #     rhs = [filter(lambda x: (type(x) == nltk.grammar.Nonterminal) and (str(x) != 'None'),
    #                   grammar.grammar.productions()[i].rhs())
    #            for i in sampled_output]
    #     for i in range(S.shape[0]):
    #         S[i].extend(map(str,rhs[i])[::-1])

    # return X_hat
    for t in range(unmasked.shape[0]):
        next_nonterminal = grammar.lhs_map[pop_or_nothing(S)]
        mask = grammar.masks[next_nonterminal]
        masked_logit = np.exp(unmasked[t])*mask  + eps
        gumbel = np.random.gumbel(size=masked_logit.shape)
        sampled_output = np.array(np.argmax(np.log(masked_logit)+gumbel,axis=-1),dtype='int')
        X_hat[t,sampled_output] = 1.0
        rhs = filter(lambda x: (type(x) == nltk.grammar.Nonterminal) and (str(x) != 'None'), grammar.grammar.productions()[sampled_output].rhs())
        push = list(reversed(list(map(str,rhs))))
        S.extend(push) #Add nonterminals to stack from right to left
    return X_hat


def prods_to_decoding(production_seq):
    """
    Takes decoded production sequence
    and returns discrete output
    """
    seq = [production_seq[0].lhs()]
    for prod in production_seq:
        if str(prod.lhs()) == 'Nothing':
            break
        for i,e in enumerate(seq):
            if e == prod.lhs():
                seq = seq[:i] + list(prod.rhs()) + seq[i+1:]
                break
    try:
        return ''.join(seq)
    except TypeError:
        return ''
