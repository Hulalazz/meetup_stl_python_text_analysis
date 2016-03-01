# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:12:50 2015

@author: Matt
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sklearn.metrics.pairwise
from sklearn.custer import KMeans
from collections import Counter


class DocumentComparer(object):
    """Class to encapsulate the Bag of Words technique for document comparison 
    and clustering. It uses scikit-learn libraries for various functions. It
    exposes the raw results as well as several custom convenience funcitons.

    Several function return lists of tuples, 
    e.g. [(str, str, num), (str, str, num), ...]
    To extract the results as columns instead, use Python's built-in zip function.
    Example:
    function_results = get_distances('document title')
    col_1, col_2, col_3 = zip(*function_results)
    """
    
    def __init__(self, *args, **kwargs):
        """
        The DocumentComparer object can be constructed with the following 
        positional arguments:
        
            DocumentComparer(docs)
            DocumentComparer(docs, names)
            DocumentComparer(docs_dict)
        
        Parameters
        ----------
        *args
            docs : list
                List of strings where each string is a document.
            names : list
                List of names for documents in docs list. Matched by index to docs.
                If names is None, the doucments will be named 0 to n-1.
            docs_dict : dict
                Dictionary of name -> document.
        """
        if len(args) == 1 and isinstance(args[0], list):
            self.docs = args[0]
            self.names = range(len(self.docs))
        elif (len(args) == 2 and isinstance(args[0], list)) and isinstance(args[1], list):
            self.docs = args[0]
            self.names = args[1]
        elif len(args) == 1 and isinstance(args[0], dict):
            self.docs = args[0].values()
            self.names = args[0].keys()
        
        # Create a dictionary of name -> index.
        self._name_idx = dict(zip(self.names, xrange(len(self.names))))
        
        # Save number of documents.
        self._n = len(self.docs)
        
    #--------------------------------------------------------------------------
    # Convenience methods
    #--------------------------------------------------------------------------
    
    def get_doc(self, name):
        """Get the text of a document by name or list of names.
        
        If a list of names is supplied, a dict will be returned.
        """
        if isinstance(name, list):
            return {n:self.docs[self._name_idx[n]] for n in name}
        else:
            return self.docs[self._name_idx[name]]
            
    #--------------------------------------------------------------------------
    # Tokenizing and Weighting
    #--------------------------------------------------------------------------
    
    def create_weighted_docs_mat(self, **kwargs):
        """This functions uses scikit-learn's CountVectorizer and 
        TfidfTransformer classes.
        
        Fit and transform the text:
        -Tokenizes each document.
        -Ignore English stopwords.
        -Builds vocabulary from words that occure in at least min_df (default 1)
            number of documents.
        -Vecorizes each consequence into an integer array.
        -Normalizes each vector using Tf-Idf.
        -Returns a matrix of num_of_docs x num_of_words (note: number of words
            are just those that make it into the vocabulary, 
            i.e. that fit teh min_df restriction above.)
        
        All kwargs are passed to the CountVectorizer constructor.
        Useful parameters include:
        min_df : integer, default 1
            Minimum number of occurences of a word over all documents in order 
            for that word to be included in the vocabulary.
        ngram_range : tuple, default (1,1)
            Range for number of words to make ngrams out of.
        token_pattern : string, default u'(?u)\\b\\w\\w+\\b'
            Regex pattern used to tokenize documents into words.
        """
        
        self.count_vectorizer = CountVectorizer(*kwargs)
        self.tfidf_transformer = TfidfTransformer()
        
        # Create a count matrix.
        self.doc_term_count_mat = self.count_vectorizer.fit_transform(self.docs)
        # Reweight it using Tf-idf
        self.weight_docs_mat = self.tfidf_transformer(self.doc_term_count_mat)
        
    
    def get_docs_containing_word(self, word):
        """Return list of (doc_name, count) tuples."""
        
        # Check if the vectorizer has been computed.
        if not hasattr(self, 'count_vectorizer'):
            raise ValueError('Count Vectorizer has not been created.')
        counts = map(lambda x: int(x[0]), self.doc_term_count_mat[:, 
                     self.count_vectorizer.vocabulary_[word]].toarry())
        tups = zip(self.names, counts)
        return filter(lambda tup: tup[1]>0, tups)
        
        
    def get_word_counts(self):
        """Returns a dict of word -> count of docs containing word."""
        
        # Check if the count_vectorizer has be computed.
        if not hasattr(self, 'count_vectorizer'):
            raise ValueError('Count Vectorizer has not been created.')
        word_included = self.doc_term_count_mat.copy()
        word_included[word_included > 0] = 1
        word_counts = np.sum(word_included.todense(), axis=0)
        return dict(zip(self.count_vectorizer.get_feature_names(), map(int, word_counts.tolist()[0])))
        
        
    def get_vocabulary(self):
        """Get the list of vocabulary words extracted from the documents."""
        
        # Check if the count_vectorizer has be computed.
        if not hasattr(self, 'count_vectorizer'):
            raise ValueError('Count Vectorizer has not been created.')
        return self.count_vectorizer.vocabulary_.keys()

        
    def get_word_weights(self):
        # Check if the count_vectorizer has be computed.
        if not hasattr(self, 'count_vectorizer'):
            raise ValueError('Count Vectorizer has not been created.')
        return sorted(zip(self.count_vectorizer.get_feature_names(), self.tfidf_transformer.idf_), 
               key=lambda tup: tup[1], reverse=True)

    #--------------------------------------------------------------------------
    # Document Distances
    #--------------------------------------------------------------------------

    def compute_distance_matrix(self):
        """Computes the Euclidean distances between documents."""
        # Check if the count_vectorizer has be computed.
        if not hasattr(self, 'count_vectorizer'):
            raise ValueError('Count Vectorizer has not been created.')
        # Create a distance matrix using a pairwise metric.
        self.dist_mat = sklearn.metrics.pairwise.pairwise_distances(self.weight_docs_mat)
        
        # Was using .eclidean_distances, but swtich to pairwise_distance 
        # because it supportes mutli-core processing.
        
        
    def get_distance_pairs(self, num=10, filter_zeros=True, zero_tol=.0001):
        """Returns num closest distances as a list of tuples of the form
        (doc_A_name, doc_B_name, distance).
       
        Skips any 0 distances.
       
        Parameters
        ----------
        num : int, default 10
            Number of pairs to return.
        fitler_zeros : bool, default True
            Ignores 0 distances when gathering results.
        zero_tol : float, default .0001
            All distance less than this number will be treated as zero.
        """
        if filter_zeros:
            self.dist_mat[self.dist_mat < zero_tol] = np.inf
        else:
            self.dist_mat[np.diag_indices(self._n)]  = np.inf
        # TODO: Just get upper triangular indices since distance matrix is symmetric.
        unr_index = np.unravel_index(self.dist_mat.argpartition(num, axis=None),
                                     self.dist_mat.shape)
        # TODO: Not sure the diagonal filter needs to be there since diagonal 
        # entries are set to inf.
        dist_pairs = [(self.names[unr_index[0][i]],        # First doc
                       self.names[unr_index[1][i]],        # Second doc
                       self.dist_mat[unr_index[0][i], unr_index[1][i]])   # Distance between docs
                       for i in xrange(num)                # First num indices
                       if self.names[unr_index[0][i] != self.names[unr_index[1][i]]]]  # Filter out diagonal entries.
        self.dist_mat[self.dist_mat == np.inf] = 0
        return sorted(dist_pairs, key=lambda tup: tup[2])
        
    
    def get_distance(self, doc_A_name, doc_B_name):
        """Returns the distance between doc_A and doc_B in the form
        (doc_A_name, doc_B_name, distance).
        """
        # Check if the distance matrix has been created.
        if not hasattr(self, 'dist_mat'):
            raise ValueError('Distance matrix has not been created.')
        return self.dist_mat[self._name_idx[doc_A_name], self._name_idx[doc_B_name]]
        
        
    def get_distances(self, doc_A_name):
        """Returns all distances between doc_A and each other document as a list
        of tuples in the form (doc_A_name, doc_B_name, distance).
        """
        # Check if the distance matrix has been created.
        if not hasattr(self, 'dist_mat'):
            raise ValueError('Distance matrix has not been created.')
        # Get the column index of doc_A.
        col = self._name_idx[doc_A_name]
        return sorted(zip([doc_A_name] * self._n, self.names, self.dist_mat[:, col]),
                      key=lambda tup: tup[2])
        
        
    def get_closest(self, doc_A_name, n=10):
        """Returns n closest documents to doc_A."""
        distances = self.get_distances(doc_A_name)
        distances.sort(key=lambda tup: tup[2])   # TODO: use argpartition
        return distances[:n]
        
    
    def search(self, new_doc):
        """Search for the closest matches in the current texts to a new 
        document. This does not add the new document to the distance matrix.
        This uses the vocabulary created with the original documents, thus
        any new vocabulary words will be ignored.
        
        Parameters
        ----------
        new_doc : string
            Text of new documet to compare to existing documents.
        """
        # Create a new document vector from the new document.
        new_doc_count_vector = self.count_vectorizer.transform([new_doc])
        new_doc_vector = self.tfidf_transformer.transform(new_doc_count_vector)        

        # Compute the distances between the new documents and existing documents.
        dists = sklearn.metrics.pairwise.euclidean_distances(new_doc_vector, 
                                                             self.weight_docs_mat)        
        # Add labels, sort, and return.
        return sorted(zip(self.names, dists[0]), key=lambda tup:tup[1])
        
    #--------------------------------------------------------------------------
    # KMeans
    #--------------------------------------------------------------------------        
    
    def compute_kmeans(self, n_clusters=8):
        """Clusters the documents using KMeans."""
            # Check if the distance matrix has been created.
        if not hasattr(self, 'dist_mat'):
            raise ValueError('Distance matrix has not been created.')
        
        # Create a KMeans object.
        self.kmeans = KMeans(n_clusters)
        
        # Cluster the documents.
        self.kmeans.fit(self.weight_docs_mat)
        
    
    def get_clusters(self):
        """Return dict of cluster_num -> list[documents in cluster num]."""
        
        # Check if the distance matrix has been created.
        if not hasattr(self, 'kmeans'):
            raise ValueError('KMeans object has not been created.')
        
        labels_pairs = zip(self.kmeans.labels_, self.names)
        self.clusters = {num: [tup[1] for tup in labels_pairs if tup[0] == num]
                            for num in set([t[0] for t in labels_pairs])}
        return self.clusters
        
        
    def get_clusters_sizes(self):
        """Return Counter (can be used like a dict) of cluster_num -> size."""

        # Check if the distance matrix has been created.
        if not hasattr(self, 'kmeans'):
            raise ValueError('KMeans object has not been created.')
        
        return Counter(self.kmeans.labels_)


    def get_cluster_words(self, cluster, num=10):
        """Return the num highest weighted words in this cluster with their average.

        Returns a list of tuples (word, cluster average)
        These are the words corresponding to the highest numbers in the cluster
        center. They are not necessaryily the words that differentiate this
        cluster from other clusters.
        """
        center = zip(self.count_vectorizer.get_feature_names(), 
                     self.kmeans.cluster_centers_[cluster])
        return sorted(filter(lambda x: x[1]>0, center), key=lambda tup: tup[1],
                      reverse=True)[:num]
                      
                      
                      
        
        
        