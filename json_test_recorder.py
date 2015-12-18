""" 
Use this to test how function scales
JSON file is updated with the key value pairs
n:time, mem, err
d:time, mem, err
m:time, mem, err
rho:time, mem, err
so that most current results are always recorded. 

"""

import blc
import timeit
import json

n = 2
m = 3
d = 2
rho = 1

filename = "json_test_recorder.json"



""" lists of values to test """
""" QUICK TEST VALUES: 
n_list = [2,3]
m_list = [2,3]
d_list = [2,3]
rho_list = [1] #density of R
"""

n_list = [10,100, 1000, 10000]
m_list = [10,100, 1000, 10000]
d_list = [2,5,10]
rho_list = [1, 0.9] #density of R
 
 
""" VARY N """ 
f = open(filename, 'r')
data = json.load(f)
f.close()
for n in n_list:
    for m in m_list:
        for d in d_list:
            for rho in rho_list:
                tuple_key = str((m,d,rho))
                R = blc.createR(n,m,d)
                W,L = blc.sampleR(R, rho) 
                
                start_time = timeit.default_timer()
                (U,V, mem) = blc.ls(R,W,d,L)
                run_time = timeit.default_timer()-start_time
                
                err = blc.rms(R, U, V)
                
                if(data['n'].has_key(tuple_key) and data['n'][tuple_key].has_key(str(n))):#add time - m,d,n combo already present
                    data['n'][tuple_key][str(n)].append(str((run_time, mem, err)))
                elif (data['n'].has_key(tuple_key)):#add m&time - d,n already present
                    data['n'][tuple_key].update({str(n):[str((run_time, mem, err))]})
                else:#add n,d,m,time
                    data['n'].update({tuple_key:{str(n):[str((run_time, mem, err))]}})
       
f = open(filename, 'w+')
f.write(json.dumps(data))
f.close()
    

""" VARY M """

f = open(filename, 'r')
data = json.load(f)
f.close()
for m in m_list:
    for n in n_list:
        for d in d_list:
            for rho in rho_list:
                tuple_key = str((n,d,rho))
                R = blc.createR(n,m,d)
                W,L = blc.sampleR(R, rho) 
                
                start_time = timeit.default_timer()
                (U,V, mem) = blc.ls(R,W,d,L)
                run_time = timeit.default_timer()-start_time
                
                err = blc.rms(R, U, V)
                
                if(data['m'].has_key(tuple_key) and data['m'][tuple_key].has_key(str(m))):#add time - m,d,n combo already present
                    data['m'][tuple_key][str(m)].append(str((run_time, mem, err)))
                elif (data['m'].has_key(tuple_key)):#add m&time - d,n already present
                    data['m'][tuple_key].update({str(m):[str((run_time, mem, err))]})
                else:#add n,d,m,time
                    data['m'].update({tuple_key:{str(m):[str((run_time, mem, err))]}})

f = open(filename, 'w+')
f.write(json.dumps(data))
f.close()


""" VARY D """
f = open(filename, 'r')
data = json.load(f)
f.close()
for d in d_list:
    for n in n_list:
        for m in m_list:
            for rho in rho_list:
                tuple_key = str((n,m,rho))
                R = blc.createR(n,m,d)
                W,L = blc.sampleR(R, rho) 
                
                start_time = timeit.default_timer()
                (U,V, mem) = blc.ls(R,W,d,L)
                run_time = timeit.default_timer()-start_time
                
                err = blc.rms(R, U, V)
                
                if(data['d'].has_key(tuple_key) and data['d'][tuple_key].has_key(str(d))):#add time - m,d,n combo already present
                    data['d'][tuple_key][str(d)].append(str((run_time, mem, err)))
                elif (data['d'].has_key(tuple_key)):#add m&time - d,n already present
                    data['d'][tuple_key].update({str(d):[str((run_time, mem, err))]})
                else:#add n,d,m,time
                    data['d'].update({tuple_key:{str(d):[str((run_time, mem, err))]}})  
  
      
""" VARY RHO """
f = open(filename, 'r')
data = json.load(f)
f.close()
for rho in rho_list:
    for n in n_list:
        for m in m_list:
            for d in d_list:
                tuple_key = str((n,m,d))
                R = blc.createR(n,m,d)
                W,L = blc.sampleR(R, rho) 
                
                start_time = timeit.default_timer()
                (U,V, mem) = blc.ls(R,W,d,L)
                run_time = timeit.default_timer()-start_time
                
                err = blc.rms(R, U, V)
                
                if(data['rho'].has_key(tuple_key) and data['rho'][tuple_key].has_key(str(rho))):#add time - m,d,n combo already present
                    data['rho'][tuple_key][str(rho)].append(str((run_time, mem, err)))
                elif (data['rho'].has_key(tuple_key)):#add m&time - d,n already present
                    data['rho'][tuple_key].update({str(rho):[str((run_time, mem, err))]})
                else:#add n,d,m,time
                    data['rho'].update({tuple_key:{str(rho):[str((run_time, mem, err))]}})          
    
  #print data    
f = open(filename, 'w+')
f.write(json.dumps(data))
f.close()  
    
    
    
