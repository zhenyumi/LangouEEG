from numpy import *
from numpy.linalg import * 
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import ranksums
from scipy.io import savemat
from scipy.io import loadmat
from random import *
from itertools import combinations
from pylab import *

'''
Python code to compute complexity measures LZc, ACE and SCE as described in "Complexity of multi-dimensional spontaneous EEG decreases during propofol induced general anaesthesia"

Author: m.schartner@sussex.ac.uk
Date: 09.12.14

To compute the complexity meaures LZc, ACE, SCE for continuous multidimensional time series X, where rows are time series (minimum 2), and columns are observations, type the following in ipython: 
 
execfile('CompMeasures.py')
LZc(X)
ACE(X)
SCE(X)


Some functions are shared between the measures.
'''

def Pre(X):
 '''
 Detrend and normalize input data, X a multidimensional time series
 '''
 ro,co=shape(X)
 Z=zeros((ro,co))
 for i in range(ro):
  Z[i,:]=signal.detrend(X[i,:]-mean(X[i,:]), axis=0)
 return Z


##########
'''
LZc - Lempel-Ziv Complexity, column-by-column concatenation
'''
##########

def cpr(string):
 '''
 Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
 '''
 d={} 
 w = ''
 i=1
 for c in string: 
  wc = w + c
  if wc in d:
   w = wc
  else:
   d[wc]=wc
   w = c
  i+=1
 return len(d)

def str_col(X):
 '''
 Input: Continuous multidimensional time series
 Output: One string being the binarized input matrix concatenated comlumn-by-column
 '''
 ro,co=shape(X)
 TH=zeros(ro)
 M=zeros((ro,co))
 for i in range(ro):
  M[i,:]=abs(hilbert(X[i,:]))
  TH[i]=mean(M[i,:])

 s=''
 for j in xrange(co):
  for i in xrange(ro):
   if M[i,j]>TH[i]:
    s+='1'
   else:
    s+='0'

 return s

def LZc(X):
 '''
 Compute LZc and use shuffled result as normalization
 '''
 X=Pre(X)
 SC=str_col(X)
 M=list(SC)
 shuffle(M)
 w=''
 for i in range(len(M)):
  w+=M[i]
 return cpr(SC)/float(cpr(w))

##########
'''
ACE - Amplitude Coalition Entropy
'''
##########

def map2(psi):
 '''
 Bijection, mapping each binary column of binary matrix psi onto an integer.
 '''
 ro,co=shape(psi) 
 c=zeros(co)
 for t in range(co):
  for j in range(ro):
   c[t]=c[t]+psi[j,t]*(2**j)
 return c

def binTrowH(M):
 '''
 Input: Multidimensional time series M
 Output: Binarized multidimensional time series
 '''
 ro,co=shape(M)
 M2=zeros((ro,co))
 for i in range(ro):
  M2[i,:]=signal.detrend(M[i,:],axis=0)
  M2[i,:]=M2[i,:]-mean(M2[i,:])
 M3=zeros((ro,co))
 for i in range(ro):
  M2[i,:]=abs(hilbert(M2[i,:]))
  th=mean(M2[i,:]) 
  for j in range(co):
   if M2[i,j] >= th :
    M3[i,j]=1
   else:
    M3[i,j]=0 
 return M3

def entropy(string):
 '''
 Calculates the Shannon entropy of a string
 '''
 string=list(string)
 prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
 entropy = - sum([ p * log(p) / log(2.0) for p in prob ])

 return entropy


def ACE(X):
 '''
 Measure ACE, using shuffled reslut as normalization.
 '''
 X=Pre(X)
 ro,co=shape(X)
 M=binTrowH(X)
 E=entropy(map2(M))
 for i in range(ro):
  shuffle(M[i])
 Es=entropy(map2(M))
 return E/float(Es)


##########
'''
SCE - Synchrony Coalition Entropy
'''
##########

def diff2(p1,p2):
 '''
 Input: two series of phases 
 Output: synchrony time series thereof 
 '''
 d=array(abs(p1-p2))
 d2=zeros(len(d))
 for i in range(len(d)):
  if d[i]>pi:
   d[i]=2*pi-d[i]
  if d[i]<0.8:
   d2[i]=1

 return d2


def Psi(X):
 '''
 Input: Multi-dimensional time series X
 Output: Binary matrices of synchrony for each series
 '''
 X=angle(hilbert(X))
 ro,co=shape(X)
 M=zeros((ro, ro-1, co))
 
 #An array containing 'ro' arrays of shape 'ro' x 'co', i.e. being the array of synchrony series for each channel. 
 
 for i in range(ro):
  l=0
  for j in range(ro):
   if i!=j:
    M[i,l]=diff2(X[i],X[j])
    l+=1
 
 return M

def BinRan(ro,co):
 '''
 Create random binary matrix for normalization
 '''

 y=rand(ro,co)
 for i in range(ro):
  for j in range(co):
   if y[i,j]>0.5:
    y[i,j]=1
   else:
    y[i,j]=0
 return y
 
def SCE(X): 
 X=Pre(X)    
 ro,co=shape(X)
 M=Psi(X)
 ce=zeros(ro)
 norm=entropy(map2(BinRan(ro-1,co)))
 for i in range(ro):
  c=map2(M[i])
  ce[i]=entropy(c)

 return mean(ce)/norm,ce/norm














