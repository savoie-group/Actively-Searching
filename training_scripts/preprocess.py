"""
Converts a list of SMILES strings into a one-hot array
as determined by the supplied Language object
"""

import argparse,h5py,sys
import numpy as np
def main(argv):
    parser = argparse.ArgumentParser(description='Uses grammar rules to construct one-hot array of production rule from smiles string')
    parser.add_argument('-f',dest='filename',default=None,help ='File containing list of smiles strings')
    parser.add_argument('-g',dest='gram',default=None,help='Grammar object')
    parser.add_argument('-b',dest='batch_size',default=100,help='Batch size to use when processing smiles list. Default: 100')
    parser.add_argument('-l',dest='max_length',default =277,help ='Maximum length to pad one-hot arrays out to. Default: 277 (Kusner)')
    parser.add_argument('-o',dest='output',default = 'processed_data.h5',help='Name to save processed data under')
    parser.add_argument('--big',dest='big',action='store_const',const=True,default=False,help ='If flag is set, uses big data routine for preprocessing data')


    args=parser.parse_args()
    filename = args.filename
    gram = args.gram
    batch_size = int(args.batch_size)
    max_length = int(args.max_length)
    output = args.output
    big = args.big


    if not gram:
        from grammar_rules import Kgram 
        from grammar_class import Grammar
        gram = Grammar()
        gram.parse_grammar(Kgram)
    
    
    preprocess(filename,gram,batch_size,max_length,output)
        
    return
def preprocess(filename,gram,batch_size,max_length,output):
    smiles = [] 
    with open(filename,'r') as f:
        for line in f:
            smiles.append(line.strip()) 

    OH = np.zeros((len(smiles),max_length,gram.N_rules))
    for i in range(0,len(smiles),batch_size):
        print('Processing: i=[' + str(i) + ':' + str(i+batch_size) + ']')
        oh = list(map(gram.one_hot,smiles[i:i+batch_size]))
        OH[i:i+batch_size,:,:]=oh 


    h5f = h5py.File(output,'w')
    h5f.create_dataset('data',data=OH)
    h5f.close()
    
    print('Program Terminated Normally')

    return


if __name__=='__main__':
    main(sys.argv[1:])
    
