import sys, re, math, time
import numpy as np
import pickle

class DataSet(object):
  def __init__(self, fpath):


  def load_all_dataset(self, FLAGS):
    fpath = FLAGS.dataset_path

    with open(fpath+"Data_set_all.pkl", 'rb') as f:
        all_dataset = pickle.load(f, encoding='latin1')

    return all_dataset

  def load_folds(self, FLAGS):
    fpath = FLAGS.dataset_path
    print("Reading %s start" % fpath)

    with open(fpath+"train_folds.pkl", 'rb') as f:
        train_datasets = pickle.load(f, encoding='latin1')
    with open(fpath+"valid_folds.pkl", 'rb') as f:
        valid_datasets = pickle.load(f, encoding='latin1')
    with open(fpath+"test_folds.pkl", 'rb') as f:
        test_datasets = pickle.load(f, encoding='latin1')

    return train_datasets, valid_datasets, test_datasets


  def load_vectors(self, FLAGS,  with_label=True): 
    fpath = FLAGS.dataset_path	
    print("Read %s start" % fpath)

    with open(fpath+"drugs_mol2vec.pkl", 'rb') as f:
        dvec = pickle.load(f, encoding='latin1')
    with open(fpath+"proteins_protvec.pkl", 'rb') as f:
        pvec = pickle.load(f, encoding='latin1')

    drug_vector = [np.array(item) for item in dvec]
    protein_vector = [np.array(item) for item in pvec]

    DTI = pickle.load(open(fpath + "Dataset_drug_protein.pkl","rb"), encoding='latin1')
    DDI = pickle.load(open(fpath + "Dataset_drug_drug.pkl","rb"), encoding='latin1')
    DSIE = pickle.load(open(fpath + "Dataset_drug_side_effect.pkl","rb"), encoding='latin1')   
    DDIS = pickle.load(open(fpath + "Dataset_drug_disease.pkl","rb"), encoding='latin1')
    PPI = pickle.load(open(fpath + "Dataset_protein_protein.pkl","rb"), encoding='latin1')
    PSIM = pickle.load(open(fpath + "Dataset_protein_similarity.pkl","rb"), encoding='latin1')
    PDIS = pickle.load(open(fpath + "Dataset_protein_disease.pkl","rb"), encoding='latin1')

    return drug_vector, protein_vector, DTI, DDI, DSIE, DDIS, PPI, PSIM, PDIS




