"""
Creates instance of a object containing
containing a set of grammar rules  and
the associated tools required to parse
the grammar. From Kusner's original GVAE
implementation
"""

import nltk
import six
import numpy as np
import utilities as util

class Grammar():
    """
    Grammar object
    """
    def __init__(self):
        pass
    def parse_grammar(self,grammar_string):
        self.grammar_string = grammar_string
        self.grammar = nltk.CFG.fromstring(grammar_string)
        self.start_index=self.grammar.productions()[0].lhs()
        self.all_left = [x.lhs().symbol() for x in self.grammar.productions()]
        self.lhs_list = list(set(self.all_left))
        self.N_rules = len(self.grammar.productions())
        self.lhs_map = {}
        for i, lhs in enumerate(self.lhs_list):
            self.lhs_map[lhs] = i
        #Here we find an array describing the next allowed production rules from the current production rule
        #that is, the allowed mappings 

        rhs_map = [None]*self.N_rules
        count = 0 
        for p in self.grammar.productions():
            rhs_map[count]=[]
            for e in p.rhs():
                if not isinstance(e,six.string_types):
                    s = e.symbol()
                    rhs_map[count].extend(list(np.where(np.array(self.lhs_list)==s)[0]))
                                                                                        
            count+=1

        self.rhs_map = rhs_map

        #Now we create the mask array

        masks = np.zeros((len(self.lhs_list),self.N_rules))
        count = 0

        for e in self.lhs_list:
            location = np.array([p == e for p in self.all_left],dtype=int).reshape(1,-1)
            #if the current left-hand rule is equal to the current unique left hand production rule. Convert to one-hot vector w/ dtype int argument and convert to row vector
            masks[count]=location
            count +=1 
        self.masks = masks
        index_array = [] 
        
        for i in range(self.N_rules): 
            index_array.append(np.where(self.masks[:,i]==1)[0][0]) 
        self.index_of_index = np.array(index_array) 

        self.max_rhs = max([len(l) for l in self.rhs_map]) 

        #At this point, mask out any rules that will never be used if desired
        #Weve removed these elements from the grammar so no need to mask
        
#        masks[:,30]=0  #BACH -> charge class
#        masks[:,32]=0  #BACH -> class
#        masks[:,4] =0 #a_o -> B
#        masks[:,8] = 0 # a_o ->S
#        masks[:,9] = 0 # a_o ->P
#        masks[:,11] = 0 #a_o -> I
#        masks[:,

    def tree(self,smiles):
        """
        From a smiles string, generate a parse tree
        """
        tokenizer = util.get_tokenizer(self.grammar) 
        smiles_array = tokenizer(smiles)
        parser = nltk.ChartParser(self.grammar) 
        parse_tree = next(parser.parse(smiles_array))
        return parse_tree

    def sequence(self,smiles):
        """
        From a smiles string, generate the
        coresponding production sequence
        """
        parse_tree = self.tree(smiles)
        production_seq = parse_tree.productions()
        return production_seq

    def one_hot(self,smiles,max_length=277):
        """
        From a smiles string, generate the
        corresponding one hot array.
        Default max length set to 277 from
        Kusner code. 
        """
        production_map = {}
        for i,e in enumerate(self.grammar.productions()):
            production_map[e]=i 
        productions_seq = self.sequence(smiles)
        indicies = np.array([production_map[production] for production in productions_seq],dtype=int) 
        oh = np.zeros([max_length,self.N_rules])
        num_productions = len(indicies)
        oh[np.arange(num_productions),indicies] = 1 
        oh[np.arange(num_productions,max_length),-1] = 1 
        return oh
