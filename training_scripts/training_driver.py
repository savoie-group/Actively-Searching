
import argparse
import sys
import os
import h5py
import numpy as np
import math

from model_VAE import VAE
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, LearningRateScheduler
from keras import backend as K

from tensorflow import set_random_seed
np.random.seed(99)
set_random_seed(99)

import h5py
import pdb
import pickle

def main(argv):
    parser=argparse.ArgumentParser(description='Driver for training autoencoder models')
    parser.add_argument('-f',dest='features',default=None,help='File containing features to train on')
    parser.add_argument('-p',dest='properties',default=None,help='File containing property data to train on. If a list is supplied, will assume space separated string and process each')
    parser.add_argument('-e',dest='epochs',default=100,help='Number of epochs to train for')
    parser.add_argument('-d',dest='dims',default=56,help='Dimensionality of latent space')
    parser.add_argument('-g',dest='grammar',default=None,help='Name of the grammar to use')
    parser.add_argument('-b',dest='batch',default=100,help='Batch size to use')
    parser.add_argument('-c',dest='checkpoint_name',default='checkpoint',help='Name to save checkpoint files under')
    parser.add_argument('--folder',dest='folder',action='store_const',const=True,default=False,help='If this flag is set, will create new folder to save outputs under')
    parser.add_argument('-fn',dest='folder_name',default='gvae_training',help='Name of folder to save output files to')
    parser.add_argument('--save_all',dest='save_all',action='store_const',const=False,default=True,help='If this flag is set, will save all checkpoint files, not just the best')
    parser.add_argument('-lr',dest='learning_rate_list',default=None,help ='Sets learning rate based on this list')
    parser.add_argument('-r',dest='restart',default=None,help='File to reload model from')
    parser.add_argument('-sw',dest='sample_weights',default = None, help = 'Sample weights for enrichment')
    parser.add_argument('-np',dest='num_props',default = 0,help = 'Number of properties')
    parser.add_argument('-val_feat',dest='val_feat',default = None, help ='Validation data')
    parser.add_argument('-val_prop',dest='val_prop',default = None, help= 'Additional validation outputs')
    parser.add_argument('-val_split',dest='val_split',default= None, help = 'For pretraining. Percentage of compounds to withhold for validation')
    parser.add_argument('-rate',dest='rate',default=0.005,help='Initial learing rate')
    parser.add_argument('--freeze',dest='freeze',action = 'store_const',const=True,default = False,help='If flag is set, will freeze autoencoder weights and only train a beefier predictor')


    args = parser.parse_args()

    train(args)


def train(args):
    
    features = args.features
    properties = args.properties
    epochs = int(args.epochs)
    dims = int(args.dims)
    gram = args.grammar
    batch=int(args.batch)
    checkpoint_name = args.checkpoint_name
    folder = args.folder
    folder_name = args.folder_name
    save_all = args.save_all
    val_feat = args.val_feat
    val_prop = args.val_prop
    learning_rate_list = args.learning_rate_list
    restart = args.restart
    sample_weights = args.sample_weights
    num_props = int(args.num_props)
    val_split = args.val_split
    freeze = args.freeze
    rate=float(args.rate)

    if val_split:
        val_split = float(val_split)

    final_name = 'trained.h5'
    
    if properties:
        properties = properties.split(' ')
    if val_prop:
        val_prop = val_prop.split(' ')

    print('Loading inputs')

    h5_file = h5py.File(features,'r')
    data = h5_file['data'][:]
    h5_file.close()

    inputs = []
    outputs = []
    weights_list = []

    samples = data.shape[0]
    print('data shape: {}'.format(data.shape))

    if sample_weights:
        print('Running  scheme')
        for i in sample_weights.split(' '):
            with open(i,'r') as f:
                weights_list.append([int(line.strip()) for line in f])
    
    index_list = []
    for i in weights_list:
        index_list.append([j for j,e in enumerate(i) if int(e)==0])

    print(data.shape)
    inputs.append(data)
    if not freeze:
        outputs.append(data)

    removed = samples - data.shape[0]

    if properties:
        for i in range(num_props):
            h5_file = h5py.File(properties[i],'r')
            pdata = h5_file['props'][:]
            outputs.append(pdata)
            h5_file.close()


    print('{} elements removed'.format(removed))



    if freeze:
        
        if num_props > 1:
            weight_dict = {}
            for p in range(num_props):
                w_array = np.asarray(weights_list[p])
                w_array = w_array.astype('float')
                w_array[w_array == 0 ] = 1e-7
                weight_dict['performance_predictor_out_{}'.format(p)] = w_array
        else:
            weight_dict = None
    else:
        #####THIS IS NEW

        if 1:

            weight_dict = {'decoded_mean':np.asarray([100 for x in weights_list[1]])}

            for p in range(num_props):
                w_mat = np.asarray(weights_list[p])
                w_mat = w_mat.astype('float')
                print(w_mat)
                w_mat[w_mat == 0] = 1e-7
                print(w_mat)
                weight_dict['predictor_out_{}'.format(p)] = w_mat
        else:
            weight_dict = None
                

    if val_feat:
        val_in = []
        val_out = []

        h5_file = h5py.File(val_feat,'r')
        val_smiles = h5_file['data'][:]
        h5_file.close()
        val_in.append(val_smiles)
        if not freeze:
            val_out.append(val_smiles)

        try:
            for i in range(num_props):
                h5_file = h5py.File(val_prop[i],'r')
                val_props = h5_file['props'][:]
                val_out.append(val_props)
                h5_file.close()
        except:
            h5_file = h5py.File(val_prop,'r')
            val_props = h5_file['props'][:]
            val_out.append(val_props)
            h5_file.close()

        val_data = [val_in,val_out]
        val_split = None
    else:
        val_data = None


    model = VAE()
    print(weight_dict)

    if not gram:
        from grammar_rules import Kgram as g
        from grammar_class import Grammar
        q = Grammar()
        q.parse_grammar(g)
        model.load_grammar(q)



    if restart:
        print('Restarting training!')
        model.load(restart,dims=dims,n_props=num_props,freeze=freeze,lr=rate)
#        val_split = None
    else:
        print('Training model from scratch')
        model.build(dims=dims,n_props=num_props,freeze=freeze)
        

    callback_list = []

    c_filepath = checkpoint_name + '{epoch:02d}-{loss:0.2f}-{val_loss:.2f}.hdf5'
    monitor = 'val_loss'
    checkpointer = ModelCheckpoint(filepath =c_filepath,
                                   monitor = monitor,
                                   verbose = 1,
                                   save_best_only = save_all,
                                   save_weights_only=True)

    callback_list.append(checkpointer)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.5,
                                  patience = 2,
                                  min_lr = 0.000001)

    callback_list.append(reduce_lr)

    class write_lr(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open('learning_rate','a') as f:
                f.write(str(K.eval(self.model.optimizer.lr))+'\n')
    wlr = write_lr()
    callback_list.append(wlr)


    if folder:
        os.makedirs(folder_name)
        os.chdir(folder_name)
        os.makedirs('Main')
        os.chdir('Main')
        os.makedirs('checkpoints')
        os.chdir('checkpoints')

    alpha = K.variable(50.0)
        
    def sigmoid(var):
        #For training on the full dataset, use b=0.1 c=50
        b=0.1
        c=50.0
        return 50.0*K.exp(-b*var+b*c)/(1.0+K.exp(-b*var+b*c))


    class LossAnnealer(Callback):
        def __init__(self,alpha):
            self.alpha = alpha
        def on_epoch_end(self,epoch,logs={}):
            self.alpha= sigmoid(epoch)

    LA = LossAnnealer(alpha)
    callback_list.append(LA)

    history =model.autoencoder.fit(
        inputs,
        outputs,
        shuffle = True,
        nb_epoch = epochs,
        batch_size = batch,
        callbacks = callback_list,
        validation_split = val_split,
        validation_data = val_data,
        sample_weight = weight_dict
    )

    def _freeze(model):
        """Freeze model weights in every layer.
        From https://stackoverflow.com/questions/51944836/keras-load-model-valueerror-axes-dont-match-array"""
        for layer in model.layers:
            layer.trainable = False

            if isinstance(layer, models.Model):
                _freeze(layer)
    _freeze(model)
    model.save(final_name)
    with open('history','wb') as f:
        pickle.dump(history.history,f)
    print('Training complete. Script terminated normally')
    


    return

if __name__ =='__main__':
    main(sys.argv[1:])




