import numpy as np
import sys 
import numpy.random as rand 
from scipy.stats import bernoulli 
import pdb
import matplotlib.pyplot as plt 
import copy 
import pandas as pd 

def cb_frontdoor(p,qyu,qzy,N,dim): 

    if qyu == 0.5: 
        p_u = 0.5 
    else: 
        p_u = 1 - ((p-qyu)/(1-2*qyu))
    
    # pdb.set_trace()

    U = rand.binomial(1,p_u,N)

    if qyu<0:
        py_u = np.array([-qyu, 1+qyu])
    else:    
        py_u = np.array([1-qyu, qyu])
    Y = rand.binomial(1,py_u[U])

    # print(sum(Y)/len(Y))

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    Z = rand.binomial(1,pz_y[Y])

    p = sum(Z)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 
    
    # sampling U (guassian with variance 5)

    idn = list(np.where(Z==0)[0]) + list(np.where(Z==1)[0])  
    Y = Y[idn]; Z = Z[idn]; U = U[idn]

    mx = np.zeros([N,dim])
    
    x_1 = np.zeros(dim); x_1[:int(dim/2)] = 2 
    x_2 = np.zeros(dim); x_2[-int(dim/2):] = 2
    b = np.repeat(1,dim)

    sigma = 3

    for i in range(N):
        mx[i] = U[i]*x_1 + Z[i]*x_2 + b 

    var = sigma*np.eye(dim)
    X = mx + rand.multivariate_normal(np.zeros(dim),var,[N])

    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)

    ## Step 1: estimate f(u,y), f(y) and f(u|y)    
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

    # confounded data 
    X_conf = X; Y_conf = Y; Z_conf = np.expand_dims(Z, axis=1); U_conf = np.expand_dims(U, axis=1)
    labels_conf = [X_conf, Y_conf, U_conf, Z_conf]
    # labels_conf = np.array([X_conf, Y_conf, U_conf, Z_conf]).transpose(1,0)

    # unconfounded data 
    X_deconf = X[i_new]; Y_deconf = np.array(Y_new); Z_deconf = np.expand_dims(Z[i_new], axis=1); U_deconf = np.expand_dims(U[i_new], axis=1)
    labels_deconf = [X_deconf, Y_deconf, U_deconf, Z_deconf]
    # labels_deconf = np.array([X_deconf, Y_deconf, U_deconf, Z_deconf]).transpose(1,0)

    return labels_conf, labels_deconf

def cb_front_n_back(p,qyu,qzy,N,dim): 
    
    if qyu == 0.5: 
        p_u = 0.5 
    else: 
        p_u = 1 - ((p-qyu)/(1-2*qyu))

    U = rand.binomial(1,p_u,N)

    if qyu<0:
        py_u = np.array([-qyu, 1+qyu])
    else:    
        py_u = np.array([1-qyu, qyu])
    Y = rand.binomial(1,py_u[U])

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    Z = rand.binomial(1,pz_y[Y])

    p = sum(Z)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Z==0)[0]) + list(np.where(Z==1)[0])  
   
    Y = Y[idn]; Z = Z[idn]; U = U[idn]

    # Genrating the dimes X1 and X2 sampling from p(x/y,u)
    mx = np.zeros([N,dim])
    sigma = 3

    x_1 = np.zeros(dim); x_1[:int(dim/2)] = 2 
    x_2 = np.zeros(dim); x_2[-int(dim/2):] = 2
    b = np.repeat(1,dim)

    for i in range(N):
        mx[i] = U[i]*x_1 + Z[i]*x_2 + b 

    var = sigma*np.eye(dim)
    X = mx + rand.multivariate_normal(np.zeros(dim),var,[N])
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)

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

    Nyu,_,_ = np.histogram2d(Y,U,bins=[len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    pu_y_emp = np.transpose(pyu_emp)/py_emp 

    ## estimate the f(z,y,u) to get f(z/y,u)  

    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [bz, by, bu]= np.histogramdd(mat,bins=[len(zr),len(yr),len(ur)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    Nyu,_,_ = np.histogram2d(Y,U, bins= [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp 

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
        #   print(sum(w))
        w = w/sum(w) ##  to renormalise 0.99999999976 
   
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()
    
    # confounded data 
    X_conf = X; Y_conf = Y; Z_conf = np.expand_dims(Z, axis=1); U_conf = np.expand_dims(U, axis=1); 
    labels_conf = [X_conf, Y_conf, U_conf, Z_conf]

    # unconfounded data 
    X_deconf = X[i_new]; Y_deconf = np.array(Y_new); Z_deconf = np.expand_dims(Z[i_new], axis=1); U_deconf = np.expand_dims(U[i_new], axis=1) 
    labels_deconf = [X_deconf, Y_deconf, U_deconf, Z_deconf]

    # ## sanity check 
    # Nyu_de,_,_ = np.histogram2d(Y_deconf,U_deconf, bins= [len(yr),len(ur)])
    # pyu_emp_de = Nyu_de/N 
    # pu_emp_de = np.sum(pyu_emp_de, axis=0)
    # py_emp_de = np.sum(pyu_emp_de, axis=1)
    # py_u_emp_de = pyu_emp_de/pu_emp_de
    # print(f"deconf correlations p(Y/U)\n: {py_u_emp_de}")

    # mat_de = np.array([Z_deconf,Y_deconf,U_deconf]).transpose(1,0)  
    # H_de, [by, bu, bz]= np.histogramdd(mat_de,bins=[len(yr),len(ur),len(zr)])
    # iz, iy, iu = np.where(H_de)
    # pzyu_emp_de = H_de/N
    # pu_emp_de = np.sum(np.sum(pzyu_emp_de, axis=0),axis=0)
    # pz_emp_de = np.sum(np.sum(pzyu_emp_de, axis=2),axis=1)
    # py_emp_de = np.sum(np.sum(pzyu_emp_de, axis=2),axis=0)
    # pyu_emp_de = np.sum(pzyu_emp_de, axis=0)    
    # pz_yu_emp_de = pzyu_emp_de/np.expand_dims(pyu_emp_de, axis=0)
    # print(f"deconf correlations p(Z/Y,U)\n: {pz_yu_emp_de}")

    # Nyz_de,_,_ = np.histogram2d(Y_deconf,Z_deconf,bins=[len(yr),len(zr)])
    # pyz_emp_de = Nyz_de/N 
    # pz_emp_de = np.sum(pyz_emp_de, axis=0)
    # py_emp_de = np.sum(pyz_emp_de, axis=1)
    # pz_y_emp_de = np.transpose(pyz_emp_de)/py_emp_de 
    # print(f"deconf correlations p(Z/Y)\n: {pz_y_emp_de}")

    # pdb.set_trace()    
    return labels_conf, labels_deconf

def cb_par_front_n_back(p,qyu,qyv,qzy,N,dim): 

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    
    if qyu<0:
        pu_y = np.array([-qyu, 1+qyu])
        pv_y = np.array([-qyv, 1+qyv])

    else:    
        pu_y = np.array([1-qyu, qyu])
        pv_y = np.array([1-qyv, qyv])

    # sampling U (guassian with variance 5)
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    V = rand.binomial(1,pv_y[Y])
    Z = rand.binomial(1,pz_y[Y])

    p = sum(Z)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Z==0)[0]) + list(np.where(Z==1)[0])     
    Y = Y[idn]; Z = Z[idn]; U = U[idn]; V = V[idn]

    mx = np.zeros([N,dim])
    
    x_1 = np.zeros(dim); x_1[:int(dim/2)] = 2 
    x_2 = np.zeros(dim); x_2[-int(dim/2):] = 5
    x_3 = np.zeros(dim); x_3[-int(dim/2):] = 5

    b = np.repeat(1,dim)
    sigma = 3  

    for i in range(N):
        mx[i] = Z[i]*x_1 + U[i]*x_2 + V[i]*x_3 + b 

    var = sigma*np.eye(dim)
    X = mx + rand.multivariate_normal(np.zeros(dim),var,[N])
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)
    vr = np.unique(V)

    ## Step 1: estimate f(z,y), f(y) and f(z|y)

    Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1)
    pz_y_emp = np.transpose(pyz_emp)/py_emp 

    ## estimate the f(z,y,u), f(y) and f()  
    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [bz, by, bu]= np.histogramdd(mat,bins=[len(zr),len(yr),len(ur)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    Nyu,_,_ = np.histogram2d(Y,U, bins= [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp

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

    # confounded data 
    X_conf = X; Y_conf = Y; Z_conf = np.expand_dims(Z, axis=1); U_conf = np.expand_dims(U, axis=1); V_conf = V 
    labels_conf = [X_conf, Y_conf, U_conf, Z_conf, V_conf]

    # unconfounded data 
    X_deconf = X[i_new]; Y_deconf = np.array(Y_new); Z_deconf = np.expand_dims(Z[i_new], axis=1); U_deconf = np.expand_dims(U[i_new], axis=1); V_deconf = V[i_new]
    labels_deconf = [X_deconf, Y_deconf, U_deconf, Z_deconf, V_deconf]
    
    return labels_conf, labels_deconf

def cb_backdoor(p,qyu,N,dim):

    if qyu == 0.5: 
        p_u = 0.5 
    else: 
        p_u = 1 - ((p-qyu)/(1-2*qyu))

    U = rand.binomial(1,p_u,N)

    if qyu<0:
        py_u = np.array([-qyu, 1+qyu])
    else:    
        py_u = np.array([1-qyu, qyu])
    Y = rand.binomial(1,py_u[U])

    p = sum(Y)/len(Y)
    # print(p)
    Ns1 = int(N*p)
    Ns0 = N - Ns1

    idn = list(np.where(Y==0)[0]) + list(np.where(Y==1)[0])  

    Y = Y[idn]; U = U[idn]

    # Genrating the dimes X1 and X2 sampling from p(x/y,u)
    mx = np.zeros([N,dim])
    
    x_1 = np.zeros(dim); x_1[:int(dim/2)] = 2 
    x_2 = np.zeros(dim); x_2[-int(dim/2):] = 2
    b = np.repeat(1,dim)
    sigma = 3 

    for i in range(N):
        mx[i] = U[i]*x_1 + Y[i]*x_2 + b 

    var = sigma*np.eye(dim)
    X = mx + rand.multivariate_normal(np.zeros(dim),var,[N])

    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    yr = np.unique(Y)
    ur = np.unique(U)

    ## Step 1: estimate f(u,y), f(y) and f(u|y)

    Nyu,_,_ = np.histogram2d(Y,U, bins= [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp 

    ## Step 2: for each y in range of values of Y variable  

    i = np.arange(0,N) # indices 
    w = np.zeros(N) # weights for the indices 
    i_new = [] 

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w[j] = (((Y==yr[m])/py_u_emp[m,U])/N)[j]
      
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(j,size=j.shape[0],replace=True,p=w[j]))
        # Y_new += [m]*j.shape[0]

    i_new.sort()
    
    # confounded data 
    X_conf = X; Y_conf = Y; U_conf = np.expand_dims(U, axis=1)
    labels_conf = [X_conf, Y_conf, U_conf]

    # unconfounded data 
    X_deconf = X[i_new]; Y_deconf = Y[i_new]; U_deconf = np.expand_dims(U[i_new], axis=1)
    labels_deconf = [X_deconf, Y_deconf, U_deconf]
    
    return labels_conf, labels_deconf

def cb_label_flip(p,qyu,qzu0,qzu1,N,dim):

    if qyu == 0.5: 
        p_u = 0.5 
    else: 
        p_u = 1 - ((p-qyu)/(1-2*qyu))

    U = rand.binomial(1,p_u,N)

    if qyu<0:
        py_u = np.array([-qyu, 1+qyu])
    else:    
        py_u = np.array([1-qyu, qyu])
    
    Y = rand.binomial(1,py_u[U])

    pz_yu = np.array([[1-qzu0, 1-qzu1], [qzu0, qzu1]])
    Z = rand.binomial(1,pz_yu[Y,U])

    p = sum(Y)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Y==0)[0]) + list(np.where(Y==1)[0])

    Y = Y[idn]; Z = Z[idn]; U = U[idn]

    mx = np.zeros([N,dim])
    x_1 = np.zeros(dim); x_1[:int(dim/2)] = 2 
    x_2 = np.zeros(dim); x_2[-int(dim/2):] = 2
    b = np.repeat(1,dim)
    sigma = 3 

    for i in range(N):
        mx[i] = U[i]*x_1 + Y[i]*x_2 + b 
    
    var = sigma*np.eye(dim)
    X = mx + rand.multivariate_normal(np.zeros(dim),var,[N])
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int)
    U = np.array(U, dtype=int) ## to make sure that they can be used as indices in later part of the code

    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)

    Nyu,_,_ = np.histogram2d(Y,U, bins= [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp 

    Nuz,_,_ = np.histogram2d(U,Z,bins=[len(ur),len(zr)])
    puz_emp = Nuz/N 
    pz_emp = np.sum(puz_emp, axis=0)
    pu_emp = np.sum(puz_emp, axis=1) 
    pz_u_emp = np.transpose(puz_emp)/pu_emp

    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    ## sanity check  
    Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1)
    pz_y_emp = np.transpose(pyz_emp)/py_emp 

    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    w1 = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):
        j = np.where(Y==yr[m])[0]
        
        w = (((pz_yu_emp[0,yr[m],U])*(Y==yr[m])/py_u_emp[yr[m],U])/N) + (((pz_yu_emp[1,yr[m],U])*(Y==yr[m])/py_u_emp[yr[m],U])/N) 
        # w = w/sum(w)

        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    # X_conf = X; Y_conf = Y; Z_conf = np.expand_dims(Z, axis=1); U_conf = np.expand_dims(U, axis=1)
    X_conf = X; Y_conf = Y; Z_conf = np.expand_dims(Z, axis=1); U_conf = np.expand_dims(U, axis=1)
    labels_conf = [X_conf, Y_conf, U_conf, Z_conf]

    # unconfounded data 
    # X_deconf = X[i_new]; Y_deconf = np.array(Y_new); Z_deconf = np.expand_dims(Z[i_new], axis=1); U_deconf = np.expand_dims(U[i_new], axis=1)
    X_deconf = X[i_new]; Y_deconf = np.array(Y_new); Z_deconf = np.expand_dims(Z[i_new], axis=1); U_deconf = np.expand_dims(U[i_new], axis=1)
    labels_deconf = [X_deconf, Y_deconf, U_deconf, Z_deconf]

    ## sanity check 
    # Nyu_de,_,_ = np.histogram2d(Y_deconf,U_deconf, bins= [len(yr),len(ur)])
    # pyu_emp_de = Nyu_de/N 
    # pu_emp_de = np.sum(pyu_emp_de, axis=0)
    # py_emp_de = np.sum(pyu_emp_de, axis=1)
    # py_u_emp_de = pyu_emp_de/pu_emp_de
    # print(f"deconf correlations p(Y/U)\n: {py_u_emp_de}")

    # mat_de = np.array([Z_deconf,Y_deconf,U_deconf]).transpose(1,0)  
    # H_de, [by, bu, bz]= np.histogramdd(mat_de,bins=[len(yr),len(ur),len(zr)])
    # iz, iy, iu = np.where(H_de)
    # pzyu_emp_de = H_de/N
    # pu_emp_de = np.sum(np.sum(pzyu_emp_de, axis=0),axis=0)
    # pz_emp_de = np.sum(np.sum(pzyu_emp_de, axis=2),axis=1)
    # py_emp_de = np.sum(np.sum(pzyu_emp_de, axis=2),axis=0)
    # pyu_emp_de = np.sum(pzyu_emp_de, axis=0)    
    # pz_yu_emp_de = pzyu_emp_de/np.expand_dims(pyu_emp_de, axis=0)
    # print(f"deconf correlations p(Z/Y,U)\n: {pz_yu_emp_de}")

    # Nyz_de,_,_ = np.histogram2d(Y_deconf,Z_deconf,bins=[len(yr),len(zr)])
    # pyz_emp_de = Nyz_de/N 
    # pz_emp_de = np.sum(pyz_emp_de, axis=0)
    # py_emp_de = np.sum(pyz_emp_de, axis=1)
    # pz_y_emp_de = np.transpose(pyz_emp_de)/py_emp_de 
    # print(f"deconf correlations p(Z/Y)\n: {pz_y_emp_de}")

    return labels_conf, labels_deconf