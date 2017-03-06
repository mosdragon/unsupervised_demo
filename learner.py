from constants import *
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def trainNB(X, Y):
	"""
	input:
		X : a list of input features (created with preproc.feature_vector)
		Y : a list of labels

	output:
		a Naive Bayes learned, trained on X data
	"""
	nb = GaussianNB()
	
	nb.fit(X, Y)
	return nb

def trainLR(X, Y):
	lr = LogisticRegression(solver="lbfgs")
	
	lr.fit(X, Y)
	return lr


def trainMLP(X, Y):
	mlp = MLPClassifier(solver="lbfgs")
	
	mlp.fit(X, Y)
	return mlp


def accuracy(nb, X, Y):
	return nb.score(X, Y)


def get_PCA_features(X_tr, X_te, n_components=None):
	pca = PCA(n_components=n_components, svd_solver="randomized")
	pca.fit(X_tr)
	t_x_tr = pca.tranform(X_tr)
	t_x_te = pca.transform(X_te)

	return t_x_tr, t_x_te


def get_ICA_features(X_tr, X_te, n_components=None):
	ica = FastICA()
	ica.fit(X_tr)
	t_x_tr = ica.tranform(X_tr)
	t_x_te = ica.transform(X_te)

	return t_x_tr, t_x_te

def get_RP_features(X_tr, X_te, n_components=None):
	rp = GaussianRandomProjection(n_components=n_components)
	rp.fit(X_tr)
	return (rp.transform(X_tr), rp.transform(X_te))


def cluster(X_tr):
	pass