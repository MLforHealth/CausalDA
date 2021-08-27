import numpy as np
import sys 
import numpy.random as rand 
from scipy.stats import bernoulli 
import pdb
import matplotlib.pyplot as plt 
import copy 
import pandas as pd 

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def cb_backdoor(index_n_labels,p,qyu,N):

    pu_y = np.array([qyu, 1-qyu]) 

    if qyu < 0: 
        pu_y = np.array([1+qyu, -qyu]) 

    filename = np.array(index_n_labels['filename'].tolist())
    Y_all = np.array(index_n_labels['label'].tolist())
    U_all = np.array(index_n_labels['conf'].tolist())

    la_all = pd.DataFrame(data={'Y_all':Y_all, 'U_all':U_all})  
    
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    
    yr = np.unique(Y); ur = np.unique(U); 
    ur_r = np.unique(U_all); yr_r = np.unique(Y_all)
    la = pd.DataFrame(data={'Y':Y,'U':U})

    Ns = []; Ns_real = []; idn = []; idx = []
    for y in yr:
        for u in ur: 
            ns = len(la.index[(la['Y']==y) & (la['U']==u)].tolist())
            Ns.append(ns)
            idn += la.index[(la['Y']==y) & (la['U']==u)].tolist()
            Ns_real.append(len(la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()))
            idx += la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()[:ns]

    Y = Y[idn]; U = U[idn]
    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    
    ## Step 1: estimate f(u,y), f(y) and f(u|y)
    Nyu,_,_ = np.histogram2d(Y,U, bins = [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp 

    ## Step 2: for each y in range of values of Y variable  
    i = np.arange(0,len(idx)) # indices 
    w = np.zeros(len(idx)) # weights for the indices 
    i_new = [] 

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w[j] = (((Y==yr[m])/py_u_emp[m,U])/N)[j]
      
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(j,size=j.shape[0],replace=True,p=w[j]))

    i_new.sort()
    
    # Step 4: New indices for unbiased data 
    idx = np.array(idx, dtype=int)    
    idx_new = idx[i_new]
    
    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; U_conf = U
    filename_conf,Y_conf,U_conf = unison_shuffled_copies(filename_conf,Y_conf,U_conf)
    labels_conf = np.array([filename_conf, Y_conf, U_conf]).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = Y[i_new]; U_deconf = U[i_new]
    filename_deconf,Y_deconf,U_deconf = unison_shuffled_copies(filename_deconf,Y_deconf,U_deconf)
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf]).transpose(1,0)
    
    return labels_conf, labels_deconf

def cb_frontdoor(index_n_labels,p,qyu,qzy,N):

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    
    if qyu<0:
        pu_y = np.array([1+qyu, -qyu])
    else:    
        pu_y = np.array([qyu, 1-qyu])
    
    filename = np.array(index_n_labels['filename'].tolist())
    Y_all = np.array(index_n_labels['label'].tolist())
    U_all = np.array(index_n_labels['conf'].tolist())

    la_all = pd.DataFrame(data={'Y_all':Y_all, 'U_all':U_all})

    Y = rand.binomial(1,p,N)
    Z = rand.binomial(1,pz_y[Y])
    U = rand.binomial(1,pu_y[Y])

    yr = np.unique(Y); ur = np.unique(U); zr = np.unique(Z)
    ur_r = np.unique(U_all); yr_r = np.unique(Y_all)
    la = pd.DataFrame(data={'Y':Y,'U':U,'Z':Z})

    Ns = []; Ns_real = []; idn = []; idx = []
    for y in yr:
        for u in ur: 
            ## since we are sampling from z 
            ns = len(la.index[(la['Z']==y) & (la['U']==u)].tolist())
            Ns.append(ns)
            idn += la.index[(la['Z']==y) & (la['U']==u)].tolist()
            Ns_real.append(len(la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()))
            idx += la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()[:ns]

    Y = Y[idn]; U = U[idn]; Z = Z[idn]
    ## to make sure that they can be used as indices in later part of the code
    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ; Z = np.array(Z, dtype=int)

    Nyz,_,_ = np.histogram2d(Y,Z,bins= [len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1)
    pz_y_emp = np.transpose(pyz_emp)/py_emp 

    ## Step 2: for each y in range of values of Y variable  
    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w = (pz_y_emp[Z,m]/pz_y_emp[Z,Y]/N)
        
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    # Step 4: New indices for unbiased data
    idx = np.array(idx)    
    idx_new = idx[i_new]

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf]).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf]).transpose(1,0)

    ## sanity check (these distribustions should change suitabely for deconfounded data)
    # Nyu,_,_ = np.histogram2d(Y_deconf,U_deconf,bins=[len(yr),len(ur)])
    # pyu_emp = Nyu/N 
    # pu_emp = np.sum(pyu_emp, axis=0)
    # py_emp = np.sum(pyu_emp, axis=1)
    # py_u_emp = np.transpose(pyu_emp)/py_emp 

    # ## estimate f(z,y,u) to get f(z/y,u)  
    # mat = np.array([Z_deconf,Y_deconf,U_deconf]).transpose(1,0)  
    # H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    # iz, iy, iu = np.where(H)
    # pzyu_emp = H/N
    # pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    # pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    # py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    # pyu_emp = np.sum(pzyu_emp, axis=0)    
    # pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    # pdb.set_trace()
    
    return labels_conf, labels_deconf

## the case with both confounding and mediator 
def cb_front_n_back(index_n_labels,p,qyu,qzy,N):

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    
    if qyu<0:
        pu_y = np.array([1+qyu, -qyu])
    else:    
        pu_y = np.array([qyu, 1-qyu])
    
    filename = np.array(index_n_labels['filename'].tolist())
    Y_all = np.array(index_n_labels['label'].tolist())
    U_all = np.array(index_n_labels['conf'].tolist())

    la_all = pd.DataFrame(data={'Y_all':Y_all, 'U_all':U_all})

    Y = rand.binomial(1,p,N)
    Z = rand.binomial(1,pz_y[Y])
    U = rand.binomial(1,pu_y[Y])

    yr = np.unique(Y); ur = np.unique(U); zr = np.unique(Z)
    ur_r = np.unique(U_all); yr_r = np.unique(Y_all)
    la = pd.DataFrame(data={'Y':Y,'U':U,'Z':Z})

    Ns = []; Ns_real = []; idn = []; idx = []
    for y in yr:
        for u in ur: 
            ## since we are sampling from z 
            ns = len(la.index[(la['Z']==y) & (la['U']==u)].tolist())
            Ns.append(ns)
            idn += la.index[(la['Z']==y) & (la['U']==u)].tolist()
            Ns_real.append(len(la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()))
            idx += la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()[:ns]

    Y = Y[idn]; U = U[idn]; Z = Z[idn]
    ## to make sure that they can be used as indices in later part of the code
    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ; Z = np.array(Z, dtype=int)

    ## Step 1: estimate f(z,y), f(y) and f(z|y)
    Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1) 
    pz_y_emp = np.transpose(pyz_emp)/py_emp

    Nuz,_,_ = np.histogram2d(U,Z,bins=[len(ur),len(zr)])
    puz_emp = Nuz/N 
    pz_emp = np.sum(puz_emp, axis=0)
    pu_emp = np.sum(puz_emp, axis=1) 
    pz_u_emp = np.transpose(puz_emp)/pu_emp 
    
    ## Step 2: for each y in range of values of Y variable  
    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w = (pz_y_emp[Z,m]/(pz_u_emp[Z,U])/N) ## conditional distribution is done by taking samples from p(x,y,z) 
                                                                ## and normalising by p(y,u)
                                                                # 
        # print(sum(w))
        w = w/sum(w) ##  to renormalise 0.99999999976 
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf]).transpose(1,0)
    
    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf]).transpose(1,0)
    
    ## sanity check (these distribustions should change suitabely for deconfounded data)
    # Nyu,_,_ = np.histogram2d(Y_deconf,U_deconf,bins=[len(yr),len(ur)])
    # pyu_emp = Nyu/N 
    # pu_emp = np.sum(pyu_emp, axis=0)
    # py_emp = np.sum(pyu_emp, axis=1)
    # py_u_emp = np.transpose(pyu_emp)/py_emp 

    # ## estimate f(z,y,u) to get f(z/y,u)  
    # mat = np.array([Z_deconf,Y_deconf,U_deconf]).transpose(1,0)  
    # H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    # iz, iy, iu = np.where(H)
    # pzyu_emp = H/N
    # pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    # pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    # py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    # pyu_emp = np.sum(pzyu_emp, axis=0)    
    # pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)
    # pdb.set_trace()

    return labels_conf, labels_deconf

def cb_par_front_n_back(index_n_labels,p,qyu,qzy,N):

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
     
    if qyu<0:
        pv_y = np.array([-qyu, 1+qyu])
        pu_y = np.array([1+qyu, -qyu])

    else:    
        pv_y = np.array([1-qyu, qyu])
        pu_y = np.array([qyu, 1-qyu])

    filename = np.array(index_n_labels['filename'].tolist())
    Y_all = np.array(index_n_labels['label'].tolist())
    U_all = np.array(index_n_labels['conf'].tolist())

    la_all = pd.DataFrame(data={'Y_all':Y_all, 'U_all':U_all})

    Y = rand.binomial(1,p,N)
    Z = rand.binomial(1,pz_y[Y])
    U = rand.binomial(1,pu_y[Y])

    yr = np.unique(Y); ur = np.unique(U); zr = np.unique(Z)
    ur_r = np.unique(U_all); yr_r = np.unique(Y_all)
    la = pd.DataFrame(data={'Y':Y,'U':U})

    Ns = []; Ns_real = []; idn = []; idx = []
    for y in yr:
        for u in ur: 
            ## since we are sampling from z 
            ns = len(la.index[(la['Z']==y) & (la['U']==u)].tolist())
            Ns.append(ns)
            idn += la.index[(la['Z']==y) & (la['U']==u)].tolist()
            Ns_real.append(len(la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()))
            idx += la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()[:ns]

    Y = Y[idn]; U = U[idn]; Z = Z[idn]
    ## to make sure that they can be used as indices in later part of the code
    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ; Z = np.array(Z, dtype=int)

    ## Step 1: estimate f(z,y), f(y) and f(z|y)

    Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1)
    pz_y_emp = np.transpose(pyz_emp)/py_emp 

    ## estimate the f(z,y,u), f(y) and f()  
    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    # pdb.set_trace()
    
    ## Step 2: for each y in range of values of Y variable  
    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w = (pz_y_emp[Z,m]/(pzyu_emp[Z,Y,U]/pyu_emp[Y,U])/N) ## conditional distribution is done by taking samples from p(x,y,z) 
                                                                ## and normalising by p(y,u) 
        w = w/sum(w) ##  to renormalise 0.99999999976 
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U; V_conf = V 
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf, V_conf]).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]; V_deconf = V[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf, V_deconf]).transpose(1,0)
    
    return labels_conf, labels_deconf

def cb_label_flip(index_n_labels,p,qyu,qzu0,qzu1,N): 
    
    if qyu<0:
        pu_y = np.array([1+qyu, -qyu])
    else:    
        pu_y = np.array([qyu, 1-qyu])

    pd_yu = np.array([[qzu0, qzu1], [1-qzu0, 1-qzu1]])

    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])    
    D = rand.binomial(1,pd_yu[Y,U])

    filename = np.array(index_n_labels['filename'].tolist())
    Y_all = np.array(index_n_labels['label'].tolist())
    U_all = np.array(index_n_labels['conf'].tolist())

    la_all = pd.DataFrame(data={'Y_all':Y_all, 'U_all':U_all})

    yr = np.unique(Y); ur = np.unique(U); dr = np.unique(D)
    ur_r = np.unique(U_all); yr_r = np.unique(Y_all)
    la = pd.DataFrame(data={'Y':Y,'U':U,'D':D})

    Ns = []; Ns_real = []; idn = []; idx = []
    for y in yr:
        for u in ur: 
            ## since we are sampling from z 
            ns = len(la.index[(la['Y']==y) & (la['U']==u)].tolist())
            Ns.append(ns)
            idn += la.index[(la['Y']==y) & (la['U']==u)].tolist()
            Ns_real.append(len(la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()))
            idx += la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()[:ns]

    Y = Y[idn]; U = U[idn]; D = D[idn]
    ## to make sure that they can be used as indices in later part of the code
    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ; D = np.array(D, dtype=int)

    Nyu,_,_ = np.histogram2d(Y,U, bins= [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp 

    mat = np.array([D,Y,U]).transpose(1,0)  
    H, [by, bu, bd]= np.histogramdd(mat,bins=[len(yr),len(ur),len(dr)])
    idd, iy, iu = np.where(H)
    pdyu_emp = H/N
    pu_emp = np.sum(np.sum(pdyu_emp, axis=0),axis=0)
    pd_emp = np.sum(np.sum(pdyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pdyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pdyu_emp, axis=0)    
    pd_yu_emp = pdyu_emp/np.expand_dims(pyu_emp, axis=0)

    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    w1 = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):
    
        j = np.where(Y==yr[m])[0]        
        w = (((pd_yu_emp[0,yr[m],U])*(Y==yr[m])/py_u_emp[yr[m],U])/N) + (((pd_yu_emp[1,yr[m],U])*(Y==yr[m])/py_u_emp[yr[m],U])/N) 
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; D_conf = D; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf, D_conf]).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); D_deconf = D[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, D_deconf]).transpose(1,0)

    ## sanity check 
    # Nyu_de,_,_ = np.histogram2d(Y_deconf,U_deconf, bins= [len(yr),len(ur)])
    # pyu_emp_de = Nyu_de/N 
    # pu_emp_de = np.sum(pyu_emp_de, axis=0)
    # py_emp_de = np.sum(pyu_emp_de, axis=1)
    # py_u_emp_de = pyu_emp_de/pu_emp_de
    # print(f"deconf correlations p(Y/U)\n: {py_u_emp_de}")

    # mat_de = np.array([D_deconf,Y_deconf,U_deconf]).transpose(1,0)  
    # H_de, [by, bu, bd]= np.histogramdd(mat_de,bins=[len(yr),len(ur),len(dr)])
    # idd, iy, iu = np.where(H_de)
    # pdyu_emp_de = H_de/N
    # pu_emp_de = np.sum(np.sum(pdyu_emp_de, axis=0),axis=0)
    # pd_emp_de = np.sum(np.sum(pdyu_emp_de, axis=2),axis=1)
    # py_emp_de = np.sum(np.sum(pdyu_emp_de, axis=2),axis=0)
    # pyu_emp_de = np.sum(pdyu_emp_de, axis=0)    
    # pd_yu_emp_de = pdyu_emp_de/np.expand_dims(pyu_emp_de, axis=0)
    # print(f"deconf correlations p(D/Y,U)\n: {pd_yu_emp_de}")

    # Nyz,_,_ = np.histogram2d(Y,D,bins=[len(yr),len(dr)])
    # pyz_emp = Nyz/N 
    # pz_emp = np.sum(pyz_emp, axis=0)
    # py_emp = np.sum(pyz_emp, axis=1)
    # pz_y_emp = np.transpose(pyz_emp)/py_emp 
    # print(f"conf corr p(D/Y)\n: {pz_y_emp}")

    # Nyz_de,_,_ = np.histogram2d(Y_deconf,D_deconf,bins=[len(yr),len(dr)])
    # pyz_emp_de = Nyz_de/N 
    # pz_emp_de = np.sum(pyz_emp_de, axis=0)
    # py_emp_de = np.sum(pyz_emp_de, axis=1)
    # pz_y_emp_de = np.transpose(pyz_emp_de)/py_emp_de 
    # print(f"deconf corr p(D/Y)\n: {pz_y_emp_de}")

    # import pdb; pdb.set_trace()

    return labels_conf, labels_deconf















