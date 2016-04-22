# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 02:49:34 2016

@author: ale
"""

# compatibility python 3
try:
    xrange
except NameError:
    xrange = range
try: input = raw_input
except NameError: pass

import blc
import timeit
import oct2py
import numpy as np
from memory_profiler import memory_usage
import pandas as pd # http://pandas.pydata.org/pandas-docs/stable/10min.html
import matplotlib # http://pandas.pydata.org/pandas-docs/stable/visualization.html
#matplotlib.style.use('ggplot')  # sudo add-apt-repository ppa:takluyver/matplotlib-daily
import pdb
import os.path
import sys
import pickle

oc = oct2py.Oct2Py()
oc.addpath('C:\Users\meeha\Desktop\FYP\MatLabOriginalFiles')

results = pd.DataFrame(columns=['density','n','m','d','mem','time','octave mem','octave time','error','octave error','python mem'])

if os.path.isfile('results1.pkl') & os.path.isfile('status1.pkl') :
  print('Results already exists in this directory.. appending...') 
  with open('status1.pkl','rb') as f:
    start = pickle.load(f) 
  results = pd.read_pickle('results1.pkl')
  input('Press Enter to continue...')
else:
  start = 0


start = 0
for i in xrange(start,1):
  for rho in [0.8]:#,0.95,0.99]:
    for n in [10]:#list(xrange(10,100,10)) + list(xrange(200,1000,100)):
      for m in [7]:#xrange(2,10,2):
        for d in [2]:#xrange(2,10,2):
          R = blc.createR(n,m,d,p)  # generate random user-item rating matrix
          (W,L,Rsampled,Rmissing) = blc.sampleR(R,rho)
          P = blc.createP(p,n)
          Lambda = blc.createLambda(P,Rsampled)
          r = oc.minimal_als('R',Rsampled,'d',d,'maxiter',10,'fast',1,'octave',1,'p',n)
          oct_err = blc.rms(Rmissing,r.U,r.V)
          print('Iteration number',i,'n',n,'m',m,'rho',rho)
          print('r.time',r.tim)
          print('oct_err',oct_err)
          #ls no groups
          start_time = timeit.default_timer()
          (U,V, mem) = blc.ls(R,Rsampled,W,d,L,0.0001,10,Lambda)#should this be R_sampled??
          run_time = timeit.default_timer()-start_time      
          blc_err = blc.rms(Rmissing,U,V)
          
          
          #ls groups
          Rtilde = blc.createRtilde(Rsampled,P)
          Rtilde_missed = blc.createRtilde(Rmissing,P)
          Wt,Lt = blc.indexExistingValues(Lambda)
          start_time = timeit.default_timer()
          (Ut, Vt, mem) = blc.ls_groups(Rtilde, Wt, d, Lt, 0.0001,10, Lambda)
          run_time_groups = timeit.default_timer()-start_time                      
          blc_err_groups = blc.rms(Rtilde_missed,Ut,Vt)
          
          print('run time', run_time)
          print('blcerr',blc_err) 
          print('run time groups', run_time_groups)
          print('blcerr groups', blc_err_groups)
          python_mem = memory_usage((blc.ls, (R,Rsampled,W,d,L,0.0001,10,Lambda)),max_usage=True)[0]*1.049e+6  # this is the total memory used 
          print(mem/1048576.0) # this is the memory in Mbytes of the variables we decided to measure
          print('-----------')
          results = results.append(pd.DataFrame({'density':rho,'n':n,'m':m,'d':d,'mem':mem,'time':run_time,'octave mem':r.mem,'octave time':r.tim,'error':blc_err,'octave error':oct_err,'python mem':python_mem},index=[0]),ignore_index=True)
  results.to_pickle('results1.pkl')
  #print 'here'
  with open('status1.pkl', 'w') as f:
    pickle.dump(i+1, f)
  #f = open('iteration', 'w')
  #f.write(str(i+1))
  #f.flush()
  #os.fsync(f.fileno())
  #f.close()
results.groupby(['n','d','m','density']).mean() # average of multiple run of same values
results[results['m'].isin([8])] # extract one part
results[(results['m'].isin([8])) & (results['d'].isin([2])) & (results['density'].isin([0.95]))] # series varying only m for a specific value of n and d
results[results['error']>1e-10].sort_values(by=['n'])
 # worst error and sorted by n
results.sort_values(by=['time'])
results.plot(x='n',y=['mem','octave mem']) #http://pandas.pydata.org/pandas-docs/stable/visualization.html
results[(results['m'].isin([8])) & (results['d'].isin([2])) & (results['density'].isin([0.8]))].plot(x='n',y=['mem','octave mem','python mem'],logy=True)
results[(results['m'].isin([8])) & (results['d'].isin([2])) & (results['density'].isin([0.8]))].plot(x='n',y=['time','octave time'],logy=True)
results[(results['m'].isin([8])) & (results['d'].isin([2])) & (results['density'].isin([0.8]))].plot(x='n',y=['error','octave error'],logy=True)
results.boxplot(column=['time','octave time'],by=['n'])
results.boxplot(column=['mem','octave mem'],by=['n'])
results.boxplot(column=['time','octave time'],by=['m'])
results.boxplot(column=['mem','octave mem'],by=['m'])
results.boxplot(column=['time','octave time'],by=['d'])
results.boxplot(column=['mem','octave mem'],by=['d'])
results.boxplot(column=['error','octave error'],by=['n'])
results.boxplot(column=['error','octave error'],by=['m'])
results.boxplot(column=['error','octave error'],by=['d'])
results.groupby(['n']).median().plot(y=['error','octave error','mem','octave mem'],logy=True) # this is the best!
results[(results['n'].isin([900])) & ( results['m'].isin([8]))].boxplot(column=['time','octave time'],by=['d','density'])
results[(results['n'].isin([900])) & ( results['m'].isin([8]))].boxplot(column=['mem','octave mem'],by=['d','density'])
results[(results['n'].isin([900])) & ( results['m'].isin([8]))].boxplot(column=['error','octave error'],by=['d','density'],sym='')
#'and results = pd.read_pickle('results.pkl') to load

# the following is even faster, ignore for now
#store = HDFStore('experiment.h5')
#store['results'] = results # save 
# and store['results']  to load

"""
FORMATTING THINGS:
LEGEND TO SIDE -
fig.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
LEGEND ON TOP
fig.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
LEGEND SIZING
fig.legend(fontsize=10,loc='center left', labelspacing=0.2)
RELABEL 
lines, labels = fig.get_legend_handles_labels()
fig.legend(lines, nlabels, loc='best')
MAKE LABEL NOT APPEAR OUTSIDE GRAPH + LEGEND UNDER GRAPH
lgd = fig.legend(lines, nlabels, loc='upper center', bbox_to_anchor=(0.5,-0.1))#legend top - not new labels
Llgd = fig.legend(lines, nlabels, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.) # legend to side
fig.get_figure().savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')


"""
