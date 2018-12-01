#!/usr/bin/env python

from sklearn.datasets import make_classification, dump_svmlight_file

fp = open('data.txt', 'wb')
for i in range(100):
	X, y = make_classification(10000, n_features=1000)
	dump_svmlight_file(X, y, fp, zero_based=False)
	print('iter %d' % i)

fp.close()
