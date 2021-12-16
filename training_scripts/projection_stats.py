import h5py
import numpy as np
from sklearn.decomposition import PCA
import argparse
import sys
import os
import re


def main(argv):
    parser = argparse.ArgumentParser(description='Calculates projection statistics')
    parser.add_argument('-w',dest='weights',default=None,help = 'File containing autoencoder weights')
    parser.add_argument('-f',dest='features',default=None,help='Features to project into latent space')
    parser.add_argument('-nd',dest='n_dims',default=None,help='Dimensionality of latent space')
    parser.add_argument('-np',dest='n_props',default=None,help='Number of properties network was trained to predict')
    parser.add_argument('--recursive',dest='recursive',action='store_const',const=True,default=False,help='If flag is set, will search through "checkpoints" folder for weights and generate statistics for each of those. Stats will be staved to "stats" folder in PCA')
    ### Output name will default to the checkpoint name + stats

    args = parser.parse_args()    
    weights = args.weights
    features = args.features
    n_dims = int(args.n_dims)
    n_props = int(args.n_props)
    recursive = args.recursive
    #Need to build encoder here
    from grammar_rules import Kgram
    from grammar_class import Grammar
    gram = Grammar()
    gram.parse_grammar(Kgram)
    from model_VAE import VAE
    gVAE = VAE()
    gVAE.load_grammar(gram)
    ###

    #Add recursive stuff
    if recursive:
#        files =[ 'checkpoints/'+f for f in os.listdir('checkpoints/') if os.path.isfile('checkpoints/'+f) and 'checkpoint' in f]
        files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.h5')]
        files = natural_sort(files)
        for f in files:
            print('Operating on {}'.format(f))
            location = f.split('.h5')[0]
            gVAE.load(weights = f,dims=n_dims,n_props=n_props)
            stats(gVAE,features,location+'_stats.h5')
    else:
        location = weights.split('.h5')[0]
        print('Loading {}'.format(location))
        gVAE.load(weights = weights,dims = n_dims, n_props= n_props)
        stats(gVAE,features,location+'_stats.h5')
    print('Script terminated normally')
    quit()

    


def stats(VAE,X,output_name):
    """
    Write description at some point
    """

    print('Reading features')
    feature_file = h5py.File(X,'r')
    data = feature_file['data'][:]
    feature_file.close()
    
    print('Generating projections')

    projections = []

    for i in range(data.shape[0]):
        latent_vector = VAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(latent_vector)

    projections = np.asarray(projections)

    print(projections.shape)
    projections = np.squeeze(projections)


    mean = np.mean(projections, axis =0)
    std = np.std(projections,axis=0)

    pca = PCA(n_components = 56, svd_solver = 'full')

    standardized_vectors= (projections - mean)/std

    
    pc_projections = pca.fit_transform(standardized_vectors)

    axes = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    
    hf5 = h5py.File(output_name,'w')
    hf5.create_dataset('projections',data = pc_projections)
    hf5.create_dataset('latent_vectors',data=projections)
    hf5.create_dataset('mean',data = mean)
    hf5.create_dataset('std',data = std)
    hf5.create_dataset('pc',data = axes)
    hf5.create_dataset('ratio',data = explained_variance_ratio)
    hf5.create_dataset('data_variance',data = pca.singular_values_)
    hf5.close()
    
#    print('Script terminated normally')
    return

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

if __name__ == '__main__':
    main(sys.argv[1:])
    
