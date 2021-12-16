import numpy as np
import time
import subprocess
import sys
import argparse
import h5py
import os
from collections import Counter
from sklearn.decomposition import PCA
import math
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import json
import joblib

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

#import utilities

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from rdkit import Chem

from scipy.spatial import ConvexHull#, Delauney
from scipy.optimize import linprog

from sklearn import linear_model

def main(argv):
    parser = argparse.ArgumentParser(description = 'Conducts reconstruction test  on latent space')
    parser.add_argument('-w',dest='weights',default = None,help ='Model to load')
    parser.add_argument('-s',dest='stats',default=None,help='Name of the files with regression info. Do not include the extensions here')
    parser.add_argument('-t',dest='test',default=None,help='File containing test smiles to encode and decode')
    parser.add_argument('-tl',dest='smile_list',default =None, help = 'File containing smiles strings')
    parser.add_argument('-n',dest='num_points',default=1000,help='Number of points to sample. Default = 1000')
    parser.add_argument('-r',dest='repeat_num',default = 1,help = 'If greater than 1, will repeatedly sample latent space and return average results')
    parser.add_argument('-np',dest='num_props',default=0,help='Number of properties network was trained to predict')
    parser.add_argument('-m',dest='method',default ='recon',help='Method to use for sampling')
    parser.add_argument('-ht',dest='high_target',default=None,help='High value(s) to target, Supply a space separated string (e.g. "0 1" The quotes must be there')
    parser.add_argument('-l',dest='low_target',default=None,help='Low value(s) to target. Supply a space separated string (e.g. "0 1" The quotes must be there')
    parser.add_argument('-tf',dest='training_features',default=None,help='Training features. You need this to perform the regression in targeted searches')
    parser.add_argument('-tp',dest='training_properties',default=None,help='Training properties. You also need this to perform the target search regression. Supply as space separated string')
    parser.add_argument('-a',dest='axis',default=None,help='Which axis to regress along. Remember to use 0-indexing. Supply a space separated string (e.g. "0 1" The quotes must be there')
    parser.add_argument('-o',dest='output_name',default=None,help='If supplied, will overwrite the name of the outputs')
    parser.add_argument('-p',dest='planes',default = None,help='Which planes to conduct the rotation on. You should supply a space separated string of three numbers denoting the planes. Something like "012 012 012" which will be interpreted as [0,1,2] (three times). Will not work correctly if you are selecting a dimension greater than 9')



    args = parser.parse_args()
    method = str(args.method)
    
    method_dict = {
        'recon':recon,
        'pc_normal':pc_normal_sample,
        'sobol':sobol_sample,
        'random':random_normal_sample,
        'c_search_1':c_search_1,
        'c_search_2':c_search_2,
        'c_search_3':c_search_3,
        'hi_u0_search':hi_u0_search,
        'special_search':special_search
        }


    method_dict[method](args)

    print('Decodings written to file. Script terminated normally\n')

    quit()


def sobol_sample(args):
    weights = args.weights
    num_points = int(args.num_points)
    repeat_num = int(args.repeat_num)


    print('Reading latent vectors')
    num_props=1
    n_dims = 2
    training_points = '/scratch/halstead/n/niovanac/recon/validity_test/data/train/smiles.h5'

    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
        
    print('Performing principal component analysis',flush=True)

    feature_file = h5py.File(training_points,'r')
    data = feature_file['data'][:]
    feature_file.close()
    
    projections = []
    
    for i in range(data.shape[0]):
        encoding = gVAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(encoding)
        
    projections = np.asarray(projections)
    latent_vectors = np.squeeze(projections)

    n_dims = latent_vectors.shape[1]
    
    if n_dims == 1:
        print('Generating Convex Hull')
        hull = get_hull(latent_vectors)
    else:
        print('Dims > 1, convex hull will not be generated')


    print('Checking decoding accuracy from prior')
    print('Generating Sobol sequence')
    
    subprocess.call(['~/SOBOL/sobol {} {} ~/SOBOL/new-joe-kuo-6.21201 > sobol_out_{}_{}.txt'.format(num_points,n_dims,n_dims,weights.split('.')[0])],shell=True)
    sobol_sequence = np.genfromtxt('sobol_out_{}_{}.txt'.format(n_dims,weights.split('.')[0]))

    max_vals = np.max(latent_vectors,axis=0)
    min_vals = np.min(latent_vectors,axis=0)


    print(max_vals)
    print(min_vals)

    rng = max_vals-min_vals

   
    scaled_sobol = (max_vals-min_vals)*(sobol_sequence)+min_vals

    np.savetxt('scaled_sobol_{}_{}.txt'.format(n_dims,weights.split('.')[0]),scaled_sobol)
    
    print('Checking if points are within convex hull')

    if n_dims == 1:
        sobol_hull = check_hull(scaled_sobol,hull)
    else:
        print(scaled_sobol.shape)
        sobol_hull = [ishull(latent_vectors,e,i) for i,e in enumerate(scaled_sobol)]
        print('\nDone')


    print('Points within hull: {}'.format(sum(sobol_hull)))
    print('Points outside of hull: {}\n'.format(num_points-sum(sobol_hull)))


    valid = 0

    pts = []
    decs = []
#    validity = []
    within = []
    without = []

    validity = np.zeros([num_points,repeat_num])


    for i,e in enumerate(scaled_sobol):
        print('{}/{} Testing point {}'.format(i+1,num_points,e))


        if sobol_hull[i]:
            hull_message = 'Inside hull'
            hull_flag = True
        else:
            hull_message = 'Outside of hull'
            hull_flag = False

        idec = []
        pts.append(e)
        for j in range(repeat_num):
#            print('Iteration {}/{}'.format(j,repeat_num),end="\r",flush=True)
            decoding = np.squeeze(gVAE.decode(np.expand_dims(e,axis=0)))
            with suppress_stdout_stderr():
                m = Chem.MolFromSmiles(str(decoding))
            if m and str(decoding)!='':
                valid = True
                validity_message = 'Valid'
            else:
                valid = False
                validity_message = 'Invalid'
#            validity.append(valid)
            validity[i,j] = valid
            if hull_flag:
                within.append(valid)
            else:
                without.append(valid)

            idec.append(str(decoding))
        print('Iteration Complete',flush=True)

        mf = most_frequent(idec)

        decs.append(mf)
        
        print('Representative Decoding: {} {:.2f}% valid ({})\n'.format(decoding,np.mean(validity[i])*100.0,hull_message))



    within = np.asarray(within)
    without = np.asarray(without)


    print('{:.2f}% ({}/{}) of points within hull'.format((sum(sobol_hull)/num_points)*100.0,sum(sobol_hull),num_points))
    print('    {:.2f}% +/- {:.2f}% reconstruction validity'.format(np.mean(within)*100,np.std(within)/(sum(within)**0.5)*100))

    print('{:.2f}% ({}/{}) of points outside of hull'.format((num_points-sum(sobol_hull))/num_points*100,num_points-sum(sobol_hull),num_points))
    print('    {:.2f}% +/- {:.2f}% reconstruction validity'.format(np.mean(without)*100,np.std(without)*100.0/(sum(without)**0.5)))

    print('Total percent valid: {:.2f}% +/- {:.2f}%'.format(np.mean(validity)*100,np.std(validity)*100.0/(np.sum(validity)**0.5)))


    with open('decodings_{}_{}.txt'.format(n_dims,weights.split('.')[0]),'w') as f:
        for i in decs:
            f.write(str(i)+'\n')

    
    np.savetxt('validity_{}_{}.txt'.format(n_dims,weights.split('.')[0]),validity)
    np.savetxt('hull_position_{}_{}.txt'.format(n_dims,weights.split('.')[0]),sobol_hull)
    np.savetxt('sampled_points_{}_{}.txt'.format(n_dims,weights.split('.')[0]),pts)

    

def recon(args):
    
    print('Reconstruction Test')
    weights = args.weights
    test = args.test
    smile_list = args.smile_list

    num_props = int(args.num_props)
    

    print('Reading test smiles',flush=True)

    smile_file = h5py.File(test,'r')
    test_smiles = smile_file['data'][:]
    smile_file.close

    feature_smiles = []
    with open(smile_list,'r') as f:
        for line in f:
            feature_smiles.append(line.strip())

#    n_dims = latent_vectors.shape[1]
    n_dims = 56
    ######Grammar block######
    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
    ##########################



    
    n_encodings = 1
    valid=[]
    check = []
    hull = []
    for i in range(n_encodings):
        print('Encoding Iteration {}/{}'.format(i+1,n_encodings),flush=True)
        encodings = gVAE.encode(test_smiles)
#        sobol_hull = [ishull(latent_vectors,e,i) for i,e in enumerate(encodings)]


#        print('Points within hull: {}'.format(sum(sobol_hull)),flush=True)
#        print('Points outside of hull: {}\n'.format(10000-sum(sobol_hull)),flush=True)

        for j,e in enumerate(encodings):
            print('Evaluating input {}'.format(j),flush=True)
 #           hull.append(ishull(latent_vectors,e,False))
            for k in range(100):
#                print('Iteration {}/{}'.format(k+1,100),end='\r',flush=True)
                decoding = np.squeeze(gVAE.decode(np.expand_dims(e,axis=0)))
                valid.append(feature_smiles[j] == decoding)
                with suppress_stdout_stderr():
                    m = Chem.MolFromSmiles(str(decoding))
                if m and str(decoding)!='':
                    check.append(True)
                else:
                    check.append(False)
            print('Done',flush=True)
#                print('Expected {} Recieved {}'.format(feature_smiles[j],decoding))

#            print('Done. Within hull? {}'.format(hull[j]),flush=True)


    valid = np.asarray(valid)
    check = np.asarray(check)
    
    print('Test Set Reconstruction Accuray: {:.2f}% +/- {:.2f}%'.format(np.mean(valid)*100,np.std(valid)/(len(valid)**0.5)))
#    print('{:.2f}% within hull'.format(sum(hull)/len(hull)*100.0))
    print('{:.2f}% +/- {:.2f}% valid'.format(np.mean(check)*100,np.std(check)/len(check)**0.5))


def random_normal_sample(args):
    weights = args.weights
    stats = args.stats
    num_points = int(args.num_points)

    stat_name = stats.split('_stats.h5')[0]

    print('Reading latent vectors')
    
    stats_file = h5py.File(stats,'r')
    latent_vectors = stats_file['latent_vectors'][:]
    stats_file.close

    n_dims = latent_vectors.shape[1]



    training_points = args.training_features

    feature_file = h5py.File(training_points,'r')
    data = feature_file['data'][:]
    feature_file.close()

    print('Read features')
    projections = []
    
    for i in range(data.shape[0]):
        if i % 10000 == 0: 
            print('On compound {}'.format(i),flush=True)
        encoding = gVAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(encoding)
        
    #For PCA, projections need to be standardized. Subtract mean and divide by std

    projections = np.asarray(projections)
    projections = np.squeeze(projections)
    
    mean_ = np.mean(projections, axis=0)
    std_ = np.std(projections,axis =0)




    ######Grammar block######
    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=0,freeze=False)
    ##########################

    print('Checking decoding accuracy from prior')
    print('Generating testing points from normal distribution')
    
#    samples = np.random.normal(loc=0.0,scale=1.0,size=(num_points,n_dims))
#    samples = np.random.normal(loc=0.0,scale=0.1,size=(num_points,n_dims))
#    samples = np.random.normal(loc=0.0,scale=0.01,size=(num_points,n_dims))


    decodings = []
    valids = []
    uniques = set()
    
    GOT_DECS=False
    GOT_VALIDS=False

    scount = 0
    while len(uniques) < num_points:
        sample = np.random.normal(loc=mean_,scale=std_,size=n_dims)
#        print('{}/{} Testing point {}'.format(i+1,num_points,e))
        decoding = np.squeeze(gVAE.decode(np.expand_dims(sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)


    with open('normal_decodings_{}_{}.txt'.format(n_dims,stat_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('normal_valids_{}_{}.txt'.format(n_dims,stat_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('normal_uniques_{}_{}.txt'.format(n_dims,stat_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')


def c_search_1(args):
    print('Running constrained search 1. Evaluates along 2d regression')
    num_props=int(args.num_props)
    weights = args.weights
    model_name = weights.split('.h5')[0].split('/')[-1]
    num_points = int(args.num_points)

    high_target = float(args.high_target)
    low_target = float(args.low_target)

    n_dims = 56
    training_points = '/scratch/halstead/n/niovanac/recon/validity_test/data/train/smiles.h5'

    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
        
    print('Performing principal component analysis',flush=True)

    feature_file = h5py.File(training_points,'r')
    data = feature_file['data'][:]
    feature_file.close()
    
    projections = []
    
    for i in range(data.shape[0]):
        encoding = gVAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(encoding)
        
    projections = np.asarray(projections)
    projections = np.squeeze(projections)
    
    mean = np.mean(projections, axis=0)
    std = np.std(projections,axis =0)
    
    pca = PCA(n_components = n_dims,svd_solver = 'full')
    
    standardized_vectors= (projections - mean)/std
    pc_projections = pca.fit_transform(standardized_vectors)
    scale = pca.explained_variance_**0.5

    print('Got principal components',flush=True)

    print('RUNNING ON 1st and 2nd PCS')
    X = pc_projections[:,:2]
    Y = np.genfromtxt('/scratch/halstead/n/niovanac/recon/validity_test/data/train/gap_ev')

    lm = linear_model.LinearRegression()
    
    model = lm.fit(X,Y)
    
    print('#'*50+' Plane Fit Statistics '+'#'*50)
    print('R2 = {}'.format(lm.score(X,Y)))
    print('Coeff: {}'.format(lm.coef_))
    print('Intercept: {}'.format(lm.intercept_))
    print('#'*110)

    m1 = lm.coef_[0]
    m2 = lm.coef_[1]
    b = lm.intercept_


    print('Single test')
    X2 = pc_projections[:,1].reshape(-1,1)
    lmt = linear_model.LinearRegression()
    modelt = lmt.fit(X2,Y)

    print('R2 = {}'.format(lmt.score(X2,Y)))
    print('Coeff: {}'.format(lmt.coef_))
    print('Intercept: {}'.format(lmt.intercept_))
    print('#'*110)


    ######
    ######
    ######

    print('Testing U0 Regression Along PC 3',flush=True)
    Xu = pc_projections[:,2].reshape(-1,1)
    Yu = np.genfromtxt('/scratch/halstead/n/niovanac/recon/validity_test/data/train/u0')

    lmu = linear_model.LinearRegression()
    
    modelu = lmu.fit(Xu,Yu)
    
    print('#'*50+' Plane Fit Statistics '+'#'*50)
    print('R2 = {}'.format(lmu.score(Xu,Yu)))
    print('Coeff: {}'.format(lmu.coef_))
    print('Intercept: {}'.format(lmu.intercept_))
    print('#'*110)



    maximal_extent_1 = np.max(pc_projections,axis=0)[2] #Maximum value along PC2
    minimal_extent_1 = np.min(pc_projections,axis=0)[2] #Min valye along PC2

    #####
    #####
    #####
    #####







    decodings = []
    valids= []
    uniques = set()
    print('Searching for structures with gap_ev between {} and {} eV'.format(low_target,high_target))
    while len(uniques) < num_points:
        normal_sample = np.random.normal(loc=0,scale=scale,size= n_dims)
        low = (low_target - b - m1*normal_sample[0])/m2
        high = (high_target -b - m1*normal_sample[0])/m2
        normal_sample[1] = np.random.uniform(low=low,high=high)

        #####
        normal_sample[2] = 1.05*minimal_extent_1 -np.abs(normal_sample[2])
        #####

        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))

        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False
        decodings.append(decoding)
        if rem and str(decoding)!='':
            valids.append(redec)
            if redec not in uniques:
                uniques.add(redec)
        print('\nDecoding: {}\n'.format(decoding),flush=True)
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,len(decodings)),flush=True)

    print('Finished run. Writing decodings to file')

    with open('constrained_search_decodings_{}-{}_{}'.format(low_target,high_target,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')

    with open('uniques_{}-{}_{}'.format(low_target,high_target,model_name),'w') as f:
        for i in uniques:
            f.write(str(i)+'\n')
    with open('valids_{}-{}_{}'.format(low_target,high_target,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')






def c_search_2(args):
    print('Running constrained search 2. Evaluates along 1d regression')
    print('Use this search method for models trained on 1 property, or on multiple uncorrelated properties')

    #Read in the command line arguments and do any preprocessing
    num_props=int(args.num_props)
    weights = args.weights
    output_name = args.output_name
    
    if output_name:
        model_name = output_name
    else:
        model_name = weights.split('.h5')[0].split('/')[-1]
    num_points = int(args.num_points)
    high_targets = [float(x) for x in args.high_target.split(' ')]
    low_targets = [float(x) for x in args.low_target.split(' ')]
    training_points = args.training_features
    axis = [int(x) for x in args.axis.split(' ')] 

    print('N-dims fixed to 56')
    n_dims= 56


    #Not a very elegant method, but this block loads in the autoencoder and grammar objects
    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
        

    print('Performing principal component analysis',flush=True)

    #Reads in training data and projects it into the latent space of the trained model

    feature_file = h5py.File(training_points,'r')
    data = feature_file['data'][:]
    feature_file.close()
    
    print('Read features')
    projections = []
    
    for i in range(data.shape[0]):
        if i % 10000 == 0: 
            print('On compound {}'.format(i),flush=True)
        encoding = gVAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(encoding)
        
    #For PCA, projections need to be standardized. Subtract mean and divide by std

    projections = np.asarray(projections)
    projections = np.squeeze(projections)
    
    mean = np.mean(projections, axis=0)
    std = np.std(projections,axis =0)
    
    pca = PCA(n_components = n_dims,svd_solver = 'full')
    
    standardized_vectors= (projections - mean)/std
    pc_projections = pca.fit_transform(standardized_vectors)

    #The eigenvalues of the covariance matrix describe the spread or variance of the data. The latent vectors are projected onto the axes with largest variance
    # i.e. largest eigenvalues

    scale = pca.explained_variance_**0.5

    print('Got principal components',flush=True)


    #Loop over axes/targets, perform regression along principal components, and find what regions must be targeted

    high_positions = []
    low_positions = []

    properties = [np.genfromtxt(y) for y in args.training_properties.split(' ')]

    for i,a in enumerate(axis):
    
        print('Regressing along PC {}'.format(a))

        X = pc_projections[:,a]
        X = X.reshape(-1,1)
        Y = properties[i]
        
        lm = linear_model.LinearRegression()    
        model = lm.fit(X,Y)
    
        print('#'*50+' Plane Fit Statistics: Axis {} '.format(a)+'#'*50)
        print('R2 = {}'.format(lm.score(X,Y)))
        print('Coeff: {}'.format(lm.coef_))
        print('Intercept: {}'.format(lm.intercept_))
        print('#'*110)


        #Just a simple linear function. y=mx+b -> x=(y-b)/m

        m1 = lm.coef_[0]
        b = lm.intercept_
    
        low = (low_targets[i]-b)/m1
        high = (high_targets[i]-b)/m1
        
        low_positions.append(low)
        high_positions.append(high)


 
    decodings = []
    valids= []
    uniques = set() #Use a set because we only care about unique elements


    print('Searching for unique structures structures with the following specifications:')
    
    for i,a in enumerate(axis): # A little silly to use enumerate when you are looking at PC0 or 1, but its neccessary for other cases
        print('Axis {} between {} and {}'.format(a,low_positions[i],high_positions[i]))

    while len(uniques) < num_points:
        normal_sample = np.random.normal(loc=0,scale=scale,size= n_dims) #You might hear this referred to as the prior distribution
        #Strictly speaking, it should just be a normal distribution with covariance == identity matrix, but in practice it works better to actually
        # get the distribution statistics from the training data 


        #While the rest of the dimensions are normally sampled, we rtestrict our axes of choice to fall within the given bounds.
        # A uniform distribution is used since we dont want to bias these values
        for i,a in enumerate(axis):
            normal_sample[a] = np.random.uniform(low=low_positions[i],high=high_positions[i])

        test_position = normal_sample


        #Now we have to transform back out of the PC space into the real latent space
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean

        #Decode the sample and check for validity, uniqueness, etc...
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False
        decodings.append(decoding)
        if rem and str(decoding)!='':
            valids.append(redec)
            if redec not in uniques:
                uniques.add(redec)
        print('\nDecoding: {} Position: {}\n'.format(decoding,test_position[axis[0]]))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,len(decodings)),flush=True)

    print('Finished run. Writing decodings to file')


    #Write the final decodings to file

    with open('constrained_search_decodings_{}'.format(model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')

    with open('uniques_{}'.format(model_name),'w') as f:
        count = 0
        for i in uniques:
            f.write(str(i)+'\n')
            count+=1
            if count == num_points:
                break
            
    with open('valids_{}'.format(model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')


def c_search_3(args):
    print('Running constrained search 3. Evaluates along 1d regression')
    print('Use this search method for models trained on multiple correlated properties')
    #It will probably work if used on a single property, or multiple correlated properties,
    #but it will be slower and might lead to strange behavior with the optimizer

    #Read in the command line arguments and do any preprocessing
    weights = args.weights
    output_name = args.output_name
    stat_name = args.stats
    
    if output_name:
        model_name = output_name
    else:
        model_name = weights.split('.h5')[0].split('/')[-1]
    num_points = int(args.num_points)
    high_targets = [float(x) for x in args.high_target.split(' ')]
    low_targets = [float(x) for x in args.low_target.split(' ')]
#    training_points = args.training_features
    axis = [int(x) for x in args.axis.split(' ')] 
    plane_list = [list([int(y) for y in x]) for x in args.planes.split(' ')]

    print('N-dims fixed to 56')
    n_dims= 56
    n_props = len(plane_list)
    print('N-properties: {}'.format(n_props))


    #Not a very elegant method, but this block loads in the autoencoder and grammar objects
    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
        

    print('Performing principal component analysis',flush=True)

    high_positions = []
    low_positions = []

    with open(stat_name+'.json','r') as f:
        regression_stats = json.load(f)

    scale = np.asarray(regression_stats["scale"]).astype('float')
    mean = np.asarray(regression_stats["mean"]).astype('float')
    std = np.asarray(regression_stats["std"]).astype('float')

    pca = joblib.load(stat_name+'.joblib')

    rotation_matrices = []
    reverse_rotation_matrices = []
    for i,a in enumerate(axis):
    
        #Build Rotation Matrix

        alpha = regression_stats["alpha_list"][i]
        beta = regression_stats["beta_list"][i]
        gamma = regression_stats["gamma_list"][i]
        ca,cb,cg,sa,sb,sg = math.cos(alpha),math.cos(beta),math.cos(gamma),math.sin(alpha),math.sin(beta),math.sin(gamma)
        R = np.array(((ca*cb,ca*sb*sg-sa*cg,ca*sb*cg+sa*sg),(sa*cb,sa*sb*sg+ca*cg,sa*sb*cg-ca*sg),(-sb,cb*sg,cb*cg)))   
        rotation_matrices.append(R)

        #Build reverse rotation matrix
        alpha *=-1
        beta*=-1
        gamma*=-1

        ca,cb,cg,sa,sb,sg = math.cos(alpha),math.cos(beta),math.cos(gamma),math.sin(alpha),math.sin(beta),math.sin(gamma)
        minus_R = np.array(((ca*cb,ca*sb*sg-sa*cg,ca*sb*cg+sa*sg),(sa*cb,sa*sb*sg+ca*cg,sa*sb*cg-ca*sg),(-sb,cb*sg,cb*cg)))   
        reverse_rotation_matrices.append(minus_R)
        #Just a simple linear function. y=mx+b -> x=(y-b)/m

        m1 = regression_stats["m_list"][i]
        b = regression_stats["b_list"][i]

        low = (low_targets[i]-b)/m1
        high = (high_targets[i]-b)/m1
        
        low_positions.append(low)
        high_positions.append(high)


 
    decodings = []
    valids= []
    uniques = set() #Use a set because we only care about unique elements


    print('Searching for unique structures structures with the following specifications:')
    
    for i,a in enumerate(axis): # A little silly to use enumerate when you are looking at PC0 or 1, but its neccessary for other cases
        print('Axis {} between {} and {}'.format(a,low_positions[i],high_positions[i]))

    while len(uniques) < num_points:
        normal_sample = np.random.normal(loc=0,scale=scale,size= n_dims) #You might hear this referred to as the prior distribution
        #Strictly speaking, it should just be a normal distribution with covariance == identity matrix, but in practice it works better to actually
        # get the distribution statistics from the training data 


        #While the rest of the dimensions are normally sampled, we rtestrict our axes of choice to fall within the given bounds.
        # A uniform distribution is used since we dont want to bias these values

        for i,a in enumerate(axis):
            pulled_plane = normal_sample[[plane_list[i]]] #pull the correct plane from the sample vector
            rotated_plane = np.matmul(pulled_plane,rotation_matrices[i]) # rotate towards gradient direction
            rotated_plane[a] = np.random.uniform(low=low_positions[i],high=high_positions[i]) #The rotation should be towards the closest pc
            unrotated_plane = np.matmul(pulled_plane,reverse_rotation_matrices[i]) #reverse the rota
            normal_sample[[plane_list[i]]] = rotated_plane #Put plane back in

        test_position = normal_sample


        #Now we have to transform back out of the PC space into the real latent space
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean

        #Decode the sample and check for validity, uniqueness, etc...
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False
        decodings.append(decoding)
        if rem and str(decoding)!='':
            valids.append(redec)
            if redec not in uniques:
                uniques.add(redec)
        print('\nDecoding: {} Position: {}\n'.format(decoding,test_position[axis[0]]))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,len(decodings)),flush=True)

    print('Finished run. Writing decodings to file')


    #Write the final decodings to file

    with open('constrained_search_decodings_{}'.format(model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')

    with open('uniques_{}'.format(model_name),'w') as f:
        count = 0
        for i in uniques:
            f.write(str(i)+'\n')
            count+=1
            if count == num_points:
                break
            
    with open('valids_{}'.format(model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')

    


def pc_normal_sample(args):

    num_props=int(args.num_props)
    print('\nSampling along single tailed distribution at maximal extent of latent encodings\n')
    weights = args.weights
    model_name = weights.split('.h5')[0].split('/')[-1]
    num_points = int(args.num_points)
    X = args.test
    n_dims = 56
    print('N-Dims hardcoded to 56')
    ######Grammar block######
    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
    ##########################

    print('Performing principal component analysis',flush=True)

    feature_file = h5py.File(X,'r')
    data = feature_file['data'][:]
    feature_file.close()
    
    projections = []
    
    for i in range(data.shape[0]):
        encoding = gVAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(encoding)
        
    projections = np.asarray(projections)
    projections = np.squeeze(projections)
    
    mean = np.mean(projections, axis=0)
    std = np.std(projections,axis =0)
    
    pca = PCA(n_components = n_dims,svd_solver = 'full')
    
    standardized_vectors= (projections - mean)/std
    pc_projections = pca.fit_transform(standardized_vectors)
    scale = pca.explained_variance_**0.5

    maximal_extent_1 = np.max(pc_projections,axis=0)[0] #Maximum value along PC1
    minimal_extent_1 = np.min(pc_projections,axis=0)[0] #Min valye along PC1

    
    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning LHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[0] = 1.05*minimal_extent_1 -np.abs(normal_sample[0])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('lhs_0_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('lhs_0_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('lhs_0_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('lhs_0_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)
    
    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning RHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[0] = 1.05*maximal_extent_1 +np.abs(normal_sample[0])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('rhs_0_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('rhs_0_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('rhs_0_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('rhs_0_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)


####
#PC2 samplings
###


    maximal_extent_2 = np.max(pc_projections,axis=0)[1] #Maximum value along PC2
    minimal_extent_2 = np.min(pc_projections,axis=0)[1] #Min valye along PC2

    
    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning LHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[1] = 1.05*minimal_extent_2 -np.abs(normal_sample[1])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('lhs_1_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('lhs_1_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('lhs_1_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('lhs_1_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)


    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning RHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[1] = 1.05*maximal_extent_2 +np.abs(normal_sample[1])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('rhs_1_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('rhs_1_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('rhs_1_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('rhs_1_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)



###
#PC3 sampling
###

    maximal_extent_3 = np.max(pc_projections,axis=0)[2] #Maximum value along PC3
    minimal_extent_3 = np.min(pc_projections,axis=0)[2] #Min valye along PC3

    
    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning LHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[2] = 1.05*minimal_extent_3 -np.abs(normal_sample[2])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('lhs_2_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('lhs_2_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('lhs_2_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('lhs_2_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)


    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning RHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[2] = 1.05*maximal_extent_3 +np.abs(normal_sample[2])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('rhs_2_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('rhs_2_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('rhs_2_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('rhs_2_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)



###
#PC4 Sampling
###

    maximal_extent_4 = np.max(pc_projections,axis=0)[3] #Maximum value along PC3
    minimal_extent_4 = np.min(pc_projections,axis=0)[3] #Min valye along PC3

    
    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning LHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[3] = 1.05*minimal_extent_4 -np.abs(normal_sample[3])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('lhs_4_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('lhs_4_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('lhs_4_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('lhs_4_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)


    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning RHS decodings\n')
    while len(decodings) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[3] = 1.05*maximal_extent_4 +np.abs(normal_sample[3])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False

        if not GOT_DECS:
            decodings.append(decoding)
            if len(decodings) >= num_points:
                print('Got decodings. Time elapsed: {}'.format(time.time()-start))
                GOT_DECS=True
        if rem and str(decoding)!='':
            if not GOT_VALIDS:
                valids.append(decoding)
                if len(valids) >= num_points:
                    GOT_VALIDS=True
                    print('Got valid decodings. Time elapsed: {}'.format(time.time()-start))
            if redec not in uniques:
                uniques.add(redec)
        
        print('\nDecoding: {}\n'.format(decoding))
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,scount),flush=True)
        
    print('Got unique decodings. Time elaspsed: {}'.format(time.time()-start))

    with open('rhs_4_pc_normal_decodings_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')
    with open('rhs_4_pc_normal_valids_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')
    with open('rhs_4_pc_normal_uniques_{}_{}.txt'.format(n_dims,model_name),'w') as f:
        for i in list(uniques):
            f.write(str(i)+'\n')
    np.savetxt('rhs_4_pc_normal_pts_{}_{}.txt'.format(n_dims,model_name),pts)


def hi_u0_search(args):

    num_props=int(args.num_props)
    print('\nSampling along single tailed distribution at maximal extent of latent encodings\n')
    weights = args.weights
    model_name = weights.split('.h5')[0].split('/')[-1]
    num_points = int(args.num_points)

    n_dims = 56
    training_points = '/scratch/halstead/n/niovanac/recon/validity_test/data/train/smiles.h5'

    print('N-Dims hardcoded to 56',flush=True)
    ######Grammar block######
    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
    ##########################

    print('Performing principal component analysis',flush=True)

    feature_file = h5py.File(training_points,'r')
    data = feature_file['data'][:]
    feature_file.close()
    
    projections = []
    
    for i in range(data.shape[0]):
        encoding = gVAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(encoding)
        
    projections = np.asarray(projections)
    projections = np.squeeze(projections)
    
    mean = np.mean(projections, axis=0)
    std = np.std(projections,axis =0)
    
    pca = PCA(n_components = n_dims,svd_solver = 'full')
    
    standardized_vectors= (projections - mean)/std
    pc_projections = pca.fit_transform(standardized_vectors)
    scale = pca.explained_variance_**0.5

    print('Testing Regression Along PC 1',flush=True)
    X = pc_projections[:,0].reshape(-1,1)
    Y = np.genfromtxt('/scratch/halstead/n/niovanac/recon/validity_test/data/train/u0')

    lm = linear_model.LinearRegression()
    
    model = lm.fit(X,Y)
    
    print('#'*50+' Plane Fit Statistics '+'#'*50)
    print('R2 = {}'.format(lm.score(X,Y)))
    print('Coeff: {}'.format(lm.coef_))
    print('Intercept: {}'.format(lm.intercept_))
    print('#'*110)




    maximal_extent_1 = np.max(pc_projections,axis=0)[0] #Maximum value along PC1
    minimal_extent_1 = np.min(pc_projections,axis=0)[0] #Min valye along PC1

    
    decodings = []
    pts = []
    valids = []
    uniques = set()
    GOT_DECS=False
    GOT_VALIDS=False
    scount = 0


    start=time.time()
    print('*'*50)
    print('\nRunning LHS decodings\n',flush=True)
    while len(uniques) < num_points:
        normal_sample = np.random.normal(loc=0,scale = scale,size = n_dims)
        normal_sample[0] = 1.05*minimal_extent_1 -np.abs(normal_sample[0])
        pts.append(normal_sample)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))
        scount+=1
        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False


        if rem and str(decoding)!='':
            valids.append(redec)
            if redec not in uniques:
                uniques.add(redec)
        print('\nDecoding: {}\n'.format(decoding),flush=True)
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,len(decodings)),flush=True)
    print('Finished run. Writing decodings to file')

    with open('constrained_search_decodings_{}'.format(model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')

    with open('uniques_{}'.format(model_name),'w') as f:
        for i in uniques:
            f.write(str(i)+'\n')


def special_search(args):
    print('Running special search along 2d regression')
    num_props=int(args.num_props)
    weights = args.weights
    model_name = weights.split('.h5')[0].split('/')[-1]
    num_points = int(args.num_points)

    high_target = float(args.high_target)
    low_target = float(args.low_target)

    n_dims = 56
    training_points = '/scratch/halstead/n/niovanac/recon/validity_test/data/train/smiles.h5'

    with suppress_stdout_stderr():
        from grammar_rules import Kgram
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
        from model_VAE import VAE
        gVAE = VAE()
        gVAE.load_grammar(gram)
        gVAE.load(weights = weights,dims=n_dims,n_props=num_props,freeze=False)
        
    print('Performing principal component analysis',flush=True)

    feature_file = h5py.File(training_points,'r')
    data = feature_file['data'][:]
    feature_file.close()
    
    projections = []
    
    for i in range(data.shape[0]):
        encoding = gVAE.encode_MV(np.expand_dims(data[i],axis=0))
        projections.append(encoding)
        
    projections = np.asarray(projections)
    projections = np.squeeze(projections)
    
    mean = np.mean(projections, axis=0)
    std = np.std(projections,axis =0)
    
    pca = PCA(n_components = n_dims,svd_solver = 'full')
    
    standardized_vectors= (projections - mean)/std
    pc_projections = pca.fit_transform(standardized_vectors)
    scale = pca.explained_variance_**0.5

    xmax = np.max(pc_projections,axis=0)[0] #Maximum value along PC1
    ymax = np.max(pc_projections,axis=0)[1] #MAx valye along PC2
    

    print('Got principal components',flush=True)

    print('RUNNING ON 1st and 2nd PCS')
    X = pc_projections[:,:2]
    Y = np.genfromtxt('/scratch/halstead/n/niovanac/recon/validity_test/data/train/gap_ev')

    lm = linear_model.LinearRegression()
    
    model = lm.fit(X,Y)
    
    print('#'*50+' Plane Fit Statistics '+'#'*50)
    print('R2 = {}'.format(lm.score(X,Y)))
    print('Coeff: {}'.format(lm.coef_))
    print('Intercept: {}'.format(lm.intercept_))
    print('#'*110)

    m1 = lm.coef_[0]
    m2 = lm.coef_[1]
    b = lm.intercept_


    print('Single test')
    X2 = pc_projections[:,1].reshape(-1,1)
    lmt = linear_model.LinearRegression()
    modelt = lmt.fit(X2,Y)

    print('R2 = {}'.format(lmt.score(X2,Y)))
    print('Coeff: {}'.format(lmt.coef_))
    print('Intercept: {}'.format(lmt.intercept_))
    print('#'*110)


    #At xmax, find ymin and max

    ymin_at_xmax = (low_target - m1 * xmax -b) / m2
    ymax_at_xmax = (high_target - m1 * xmax -b) / m2

    #At ymax, find ymin and max

    xmin_at_ymax = (low_target - m2 * ymax -b) / m1
    xmax_at_ymax = (high_target - m2 * ymax -b) /m1


    decodings = []
    valids= []
    uniques = set()
    print('Searching for structures with gap_ev between {} and {} eV'.format(low_target,high_target))
    while len(uniques) < num_points:
        normal_sample = np.random.normal(loc=0,scale=scale,size= n_dims)
        
        #Uniformly sample X

        normal_sample[0] = np.random.uniform(low=xmin_at_ymax,high=xmax)

        #From regression, determine high and low y to uniformly sample from
        low = (low_target - b - m1*normal_sample[0])/m2
        high = (high_target -b - m1*normal_sample[0])/m2

        normal_sample[1] = np.random.uniform(low=low,high=high)
        normal_sample = (((pca.inverse_transform(normal_sample).reshape(1,-1)))[0])*std+mean
        decoding = np.squeeze(gVAE.decode(np.expand_dims(normal_sample,axis=0)))

        with suppress_stdout_stderr():
            m = Chem.MolFromSmiles(str(decoding))
            if m:                
                redec = Chem.MolToSmiles(m)
                rem = Chem.MolFromSmiles(redec)
                if not rem:
                    print('Failed to convert to canonical structure')
                    decoding = ''
            else:
                rem = False
        decodings.append(decoding)
        if rem and str(decoding)!='':
            valids.append(redec)
            if redec not in uniques:
                uniques.add(redec)
        print('\nDecoding: {}\n'.format(decoding),flush=True)
        print('{} decodings, {} valid, {} unique. {} requested, {} processed so far'.format(len(decodings),len(valids),len(uniques),num_points,len(decodings)),flush=True)

    print('Finished run. Writing decodings to file')

    with open('special_search_decodings_{}-{}_{}'.format(low_target,high_target,model_name),'w') as f:
        for i in decodings:
            f.write(str(i)+'\n')

    with open('uniques_{}-{}_{}'.format(low_target,high_target,model_name),'w') as f:
        for i in uniques:
            f.write(str(i)+'\n')
    with open('valids_{}-{}_{}'.format(low_target,high_target,model_name),'w') as f:
        for i in valids:
            f.write(str(i)+'\n')





def get_hull(latent_vectors):
    hull = ConvexHull(latent_vectors)
    return hull

def check_hull(points,hull):
    A = hull.equations[:,0:-1]
    b = np.transpose(np.array([hull.equations[:,-1]]))
    isInHull = np.all((A @ np.transpose(points)) <= np.tile(-b,(1,len(points))),axis=0)
    return isInHull


def ishull(p,q,index):
#    if index:
#        print('Testing point {}'.format(index),end="\r",flush=True)
    hull_size = p.shape[0]
    n_dims = p.shape[1]
    constraint = np.zeros(hull_size) # Minimize c @ x. Since no c, this is a feasibility test
    A = np.r_[p.T,np.ones((1,hull_size))]
    b = np.r_[q,np.ones(1)]
    lp = linprog(constraint,A_eq=A,b_eq=b,method='interior-point')
    print(lp.status)
    return lp.success


def most_frequent(List):
    List = list(filter(('').__ne__,List))
    if len(List) == 0:
        List = ['']
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]





if __name__ == '__main__':
    main(sys.argv[1:])
