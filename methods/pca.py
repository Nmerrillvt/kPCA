"""This module contains the function to perform principal components analysis."""
import numpy as np
import warnings

class PCA:
    def __init__(self,q):
        self.q = q
 
    def mean_center(self,data):
        """
        mean center the data for linear PCA
        """
        mean = np.repeat(np.expand_dims(np.mean(data,axis=0),axis=0),data.shape[0],axis=0)
        data = (data-mean)
        return(data)
    
    def check_data(self,data):
        n_samples, d_features = np.shape(data)
        
        if d_features > n_samples:
            warnings.warn("Data should be specified as (n, d) where each row is\
                          a data example and each column is a feature.")
            
        self.d_features = d_features
        self.n_samples = n_samples
        
        data = self.mean_center(data)
        
        return data
    
    
    def fit(self,data):
        """Fit detector. y is optional for unsupervised methods.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, d_features)
        The input samples.
        
        y : numpy array of shape (n_samples,), optional (default=None)
        The ground truth of the input samples (labels).
        """
          
        data = self.check_data(data)
        
        self.Covar_mat = (1/self.n_samples)*data.T.dot(data)
        
        #do eigendcomposition
        values, vectors = np.linalg.eig(self.Covar_mat)
        
        #sort in descending order
        idx = np.argsort(-np.real(values))
        self.variances = values[idx]
        self.eigenvectors = vectors[:,idx]
        

    def decision_function(self, data):
        """predict anomaly scores (reconstruction error) 
        
        Parameters
        ----------
        data : numpy array of shape (n_test_samples, d_features)
        The test samples.
        """
        
        data = self.check_data(data)
        
        if data.ndim == 1: # correct dimension if a single example is given
            data = np.expand_dims(data,axis = 0)
            
        new_data = (self.eigenvectors.T.dot(data.T)).T
        r_data = (self.eigenvectors[:,:self.q].T.dot(data.T)).T
        
        scores = new_data.dot(new_data.T).sum(axis=0) - r_data.dot(r_data.T).sum(axis=0)
        
        return scores
    