import math
import numpy as np
import scipy
from scipy.sparse import diags
from scipy import sparse
from scipy.stats import uniform
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
import scipy.linalg
from scipy.sparse import identity, issparse
import scipy.sparse.linalg as sparse_la
import time
import os.path
import bisect
import matplotlib.pyplot as plt
import sys
import argparse
import yaml
import os

##################################################################
################## GLOBAL USED FUNCTIONS #########################
##################################################################

#Returns the value of l from the state index.
def the_l(ix):
    if ix == 0:
        return 0
    elif ix % 2 != 0:
        return int((ix+1) / 2)
    else:
        return -int(ix/2)
    
#Returns the value of omega.
def omega(k, M, R):
    if not isinstance(k, int):
        raise ValueError("Error in calculating omega: k must be an integer.")
    omega = math.sqrt( (k**2 / R**2) + M**2)
    return omega

##################################################################
################## GENERATE THE BASIS OF STATES ##################
##################################################################

#Genrates the basis of states. 
#Each state is represented as |n0, n1, n2, n3, ...>
#n0 is the number of particles with l=0,
#n1 is the number of particles with l=1,
#n2 is the number of particles with l=-1,
#n3 is the number of particles with l=2,
#n4 is the number of particles with l=-2, etc.
def gen_basis(omega_list,lmax, Emax, M, R):
    w = omega_list[lmax]  # ω_l corrente

    if lmax == 0:
        max_n = int(Emax // M)
        for n in range(max_n + 1):
            yield (n,)  # singola tupla contenente solo n_0
    else:
        max_n_total = int(Emax // w)
        for n in range(max_n_total + 1):
            E_remain = Emax - n * w
            for prev in gen_basis(omega_list,lmax - 1, E_remain, M, R):
                for NP in range(n + 1):  # NP = # particelle con +l, (n-NP) = con -l
                    yield prev + (NP, n - NP)

################## FILTER THE BASIS OF STATES ####################

#Filters Z=+1 (Even number of particles) states.
def filter_even(basis):
    for state in basis:
        if sum(state) % 2 == 0:
            yield state

#Filters Z=-1 (Even number of particles) states.
def filter_odd(basis):
    for state in basis:
        if sum(state) % 2 != 0:
            yield state
            
#Filters the states with l_tot=0. All the other states can be obtained from these with an overall factor.
def filter_l0(basis):
    for state in basis:
        if l_total(state) == 0:
            yield state

#Filters the states with l_tot=l1,l2...
def filter_moments(k_list, basis):
    k_set = set(k_list)  # conversione in set per rendere il test di appartenenza più efficiente
    for state in basis:
        if l_total(state) in k_set:
            yield state


##################################################################
################## FUNCTION OF THE STATES ########################
##################################################################

#Returns the Total Energy of the state.
def state_energy(state,M,R):
    total_energy = 0
    for ix in range(len(state)):
        total_energy += state[ix] * omega(the_l(ix), M, R)
    return total_energy

#Returns total l of the state.
def l_total(state):
    l_total = 0
    for i in range(1, len(state)):
        l_total += state[i] * the_l(i)
    return l_total

#Returns the total number of particles of the state.
def n_total(state):
    n_total = 0
    for i in range(len(state)):
        n_total += state[i]
    return n_total


##################################################################
######################### H0 MATRIX  #############################
##################################################################

#Generation of H0 matrix.
def H0(basis,M,R):
    diag = [state_energy(state,M,R) for state in basis]
    H0 = diags(diag, offsets=0, format='csr') 
    return H0

##################################################################
######################### H2 MATRIX  #############################
##################################################################

#Generation of the list [k,-k] for a_k a_k.
def low2_index(omega_list,lmax,Emax,M,R):
    list_index=[]
    for k in range(0,lmax+1):
        if k == 0:
            m=1
        else:
            m=2
        if 2*omega_list[k] <= Emax: 
                index=[k,-k]
                list_index.append([index,m])
    return list_index

#Returns the matrix element of a_l a_k
def low2(basis, state_index, ix_list, l, k):
    #term: (a_l) (a_k)
    row=[]
    col=[]
    values=[]
    ix_l=ix_list[l]
    ix_k=ix_list[k]
    
    for state_ix in range(len(basis)):
        pos=None
        state = list(basis[state_ix])

        #verify n_k>0
        if state[ix_k] !=0: 
            factor1= math.sqrt(state[ix_k])
            state[ix_k] -= 1

            if state[ix_l] !=0: 
                factor2= factor1*math.sqrt(state[ix_l])
                state[ix_l] -= 1
                pos= state_index.get(tuple(state))
                
                if pos != None:
                    row.append(pos)
                    col.append(state_ix)
                    values.append(factor2)

    return row, col, values

#Evaluates the term phi+^2.
def Term_low2(basis, state_index, omega_list, ix_list, lmax, Emax, M, R):
    rows = []
    cols = []
    data = []
    N = len(basis)

    # 1) calcola l’insieme degli indici e dei moltiplicatori (come in low4_index)
    for k_tuple, molt in low2_index(omega_list, lmax, Emax, M, R):

        k1, k2 = k_tuple
        # fattore scalare esterno
        factor = molt * ( 1.0 / (2*omega_list[k1]) ) 

     # 2) estrai le entry raw
        sub_rows, sub_cols, sub_vals = low2(basis, state_index, ix_list,  k1, k2)

    # 3) accumula, moltiplicando in blocco
        rows.extend(sub_rows)
        cols.extend(sub_cols)
        data.extend([factor * v for v in sub_vals])

    # 4) un’unica creazione COO + conversione a CSR
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

#Evaluates the term phi+ phi-.
def Term_rai1_low1(basis,omega_list,M,R):
    diag = []
    for state_ix in range(len(basis)):
        state = list(basis[state_ix])
        factor=0
        for ix in range(len(state)):
            factor+= (state[ix]/(2*omega_list[the_l(ix)])) 
        diag.append(factor)
    term_rai1_low1 = diags(diag, offsets=0, format='csr') 
    return term_rai1_low1

#generation of H2 = int dx {N[phi^2]}<---- N[phi^2]= (phi-)^2 + (phi+)^2 + 2 phi- phi+
def H2(basis, state_index, omega_list, ix_list, lmax, Emax, M, R):
    term_low2 = Term_low2(basis, state_index, omega_list, ix_list, lmax, Emax, M, R)
    term_rai2 = term_low2.transpose().tocsr() #transpose of csr is csc
    term_rai1_low1 = Term_rai1_low1(basis, omega_list,M, R)
    H2 = term_rai2 + term_low2 + 2*term_rai1_low1
    return H2

##################################################################
######################### H4 MATRIX  #############################
##################################################################

######################### phi+^4   ###############################
def low4_index(omega_list, lmax,Emax,M,R):
    list_index=[]

    #lmax <= k1 < k2 < k3 < k4 <= lmax
    for k1 in range(-lmax,lmax-2):
        for k2 in range(k1+1,lmax-1):
            for k3 in range(k2+1,lmax):
                k4=-k1-k2-k3
                if k4 in range(k3+1,lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k4] <= Emax:
                    index=[k1,k2,k3,k4]
                    m=24 #4!
                    list_index.append([index,m])

    #lmax <= k1 < k2 < k3 = k4 <= lmax
    for k1 in range(-lmax,lmax-1):
        for k2 in range(k1+1,lmax):
                if (-k1-k2) % 2 == 0:
                    k3 = int((-k1-k2)/2)
                    k4= k3
                    if k4 in range(k2+1,lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k4] <= Emax:
                        index=[k1,k2,k3,k4]
                        m=12 #4!/2!
                        list_index.append([index,m])
    
    # -lmax <=  k1 < k2 = k3 < k4 <= lmax
    for k1 in range(-lmax, lmax-1):
        for k2 in range(k1+1, lmax):
            k3=k2
            k4=-k1-k2-k3
            if k4 in range(k2+1, lmax+1) and  omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k4] <= Emax:
                index=[k1,k2,k3,k4]
                m=12 #4!/2!
                list_index.append([index,m])
    
    #-lmax <= k1 = k2 < k3 < k4 <= lmax
    for k1 in range(-lmax, lmax-1):
        for k3 in range(k1+1, lmax):
            k2=k1
            k4=-k1-k2-k3
            if k4 in range(k3+1, lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k4] <= Emax:
                index=[k1,k2,k3,k4]
                m=12 #4!/2!
                list_index.append([index,m])

    #-lmax <= k1 = k2 < k3 = k4 <= lmax
    for k1 in range(-lmax, lmax):
        k2=k1
        k3=-k1
        k4=k3
        if k4 in range(k1+1, lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k4] <= Emax:
            index=[k1,k2,k3,k4]
            m=6 #4!/(2!*2!)
            list_index.append([index,m])

    #-lmax <= k1 < k2 = k3 = k4 <= lmax
    for k1 in range(-lmax, lmax):
        if k1 % 3 == 0:
            k2 = - int(k1/3)
            k3=k2
            k4=k3
            if k4 in range(k1+1, lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k4] <= Emax:
                index=[k1,k2,k3,k4]
                m=4 #4!/3!)
                list_index.append([index,m])

    #-lmax <= k1 = k2 = k3 < k4 <= lmax
    for k1 in range(-lmax, lmax):
        k2=k1
        k3=k1
        k4=-k1-k2-k3
        if k4 in range(k1+1, lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k4] <= Emax:
            index=[k1,k2,k3,k4]
            m=4 #4!/3!)
            list_index.append([index,m])

    #-lmax <= k1 = k2 = k3 = k4 <= lmax
    k1=0
    k2=0
    k3=0
    k4=0
    if omega_list[k1]+omega_list[k2]+omega_list[k3]+omega_list[k3] <= Emax:
            index=[k1,k2,k3,k4]
            m=1 #4!/4!
            list_index.append([index,m])

    return list_index

def low4(basis,state_index, ix_list, k1,k2,k3,k4):
    #term: (a_l) (a_k)
    row=[]
    col=[]
    values=[]
    ix_k1= ix_list[k1]
    ix_k2= ix_list[k2]
    ix_k3= ix_list[k3]
    ix_k4= ix_list[k4]

    for state_ix in range(len(basis)):
        pos=None
        state = list(basis[state_ix])

        #verify n_k>0
        if state[ix_k4] !=0: 
            factor1= math.sqrt(state[ix_k4])
            state[ix_k4] -= 1 
            #verify n_l>0
            if state[ix_k3] !=0: 
                factor2= factor1*math.sqrt(state[ix_k3])
               
                state[ix_k3] -= 1
                if state[ix_k2] !=0:
                    factor3= factor2*math.sqrt(state[ix_k2])
                    state[ix_k2] -= 1
                    if state[ix_k1] !=0:
                        factor4= factor3*math.sqrt(state[ix_k1])
                        state[ix_k1] -= 1

                        pos = state_index.get(tuple(state))
                        if pos != None:
                            row.append(pos)
                            col.append(state_ix)
                            values.append(factor4)

    return row, col, values

def Term_low4(basis, state_index, omega_list,  ix_list, lmax, Emax, M, R):
    rows = []
    cols = []
    data = []
    N = len(basis)

    # 1) calcola l’insieme degli indici e dei moltiplicatori (come in low4_index)
    for k_tuple, molt in low4_index(omega_list, lmax, Emax, M, R):

        k1, k2, k3, k4 = k_tuple
        # fattore scalare esterno
        factor = molt * (1.0 / (4 * math.sqrt(omega_list[k1] * omega_list[k2] * omega_list[k3] * omega_list[k4])))* (1.0 / (2 * math.pi * R))

     # 2) estrai le entry raw
        sub_rows, sub_cols, sub_vals = low4(basis, state_index, ix_list,  k1, k2, k3, k4)

    # 3) accumula, moltiplicando in blocco
        rows.extend(sub_rows)
        cols.extend(sub_cols)
        data.extend([factor * v for v in sub_vals])

    # 4) un’unica creazione COO + conversione a CSR
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

######################### phi- phi+^3   ###########################
def rai1_low3_index(omega_list,lmax,Emax,M,R):
    list_index=[]
    #lmax <= k1 < k2 < k3 <= lmax
    for k1 in range(-lmax,lmax-1):
        for k2 in range(k1+1,lmax):
            for k3 in range(k2+1,lmax+1):
                k4= k1+k2+k3
                if k4 in range(-lmax,lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]<= Emax and omega_list[k4]<= Emax:
                    index=[k1,k2,k3,k4]
                    m=6 #3!
                    list_index.append([index,m])
    
    #lmax <= k1 = k2 < k3 <= lmax
    for k1 in range(-lmax,lmax):
            for k3 in range(k1+1,lmax+1):
                k2=k1
                k4= k1+k2+k3
                if k4 in range(-lmax,lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]<= Emax and omega_list[k4]<= Emax:
                    index=[k1,k2,k3,k4]
                    m=3 #3!/2!
                    list_index.append([index,m])

    #lmax <= k1 < k2 = k3 <= lmax
    for k1 in range(-lmax,lmax):
            for k3 in range(k1+1,lmax+1):
                k2=k3
                k4= k1+k2+k3
                if k4 in range(-lmax,lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]<= Emax and omega_list[k4]<= Emax:
                    index=[k1,k2,k3,k4]
                    m=3 #3!/2!
                    list_index.append([index,m])
    
    #lmax <= k1 = k2 = k3 <= lmax
    for k1 in range(-lmax,lmax+1):
        k2=k1
        k3=k1
        k4= k1+k2+k3
        if k4 in range(-lmax,lmax+1) and omega_list[k1]+omega_list[k2]+omega_list[k3]<= Emax and omega_list[k4]<= Emax:
            index=[k1,k2,k3,k4]
            m=1 #3!/3!
            list_index.append([index,m])
    return list_index

def rai1_low3(basis,state_index, ix_list, k1,k2,k3,k4):
    #k1,k2,k3 ->a
    #k4-> a^\dagger
    row=[]
    col=[]
    values=[]
    for state_ix in range(len(basis)):
        pos=None
        state = list(basis[state_ix])
        ix_k1= ix_list[k1]
        ix_k2= ix_list[k2]
        ix_k3= ix_list[k3]
        ix_k4= ix_list[k4]
        dim= len(basis)
        #verify n_k1>0
        if state[ ix_list[k1]] !=0: 
            factor1= math.sqrt(state[ix_k1])
            state[ix_k1] -= 1 

            #verify n_k2>0
            if state[ix_k2] !=0: 
                factor2= factor1*math.sqrt(state[ix_k2])
                state[ix_k2] -= 1

                #verify n_k3>0
                if state[ix_k3] !=0:
                    factor3= factor2*math.sqrt(state[ix_k3])
                    
                    state[ix_k3] -= 1
                    
                    #now a^\dagger
                    factor4= factor3*math.sqrt(state[ix_k4]+1)
                   
                    state[ix_k4] += 1
          
                    pos = state_index.get(tuple(state))
                    if pos != None:
                        row.append(pos)
                        col.append(state_ix)
                        values.append(factor4)

    return row, col, values

def Term_rai1_low3(basis, state_index, omega_list, ix_list, lmax, Emax, M, R):
    rows = []
    cols = []
    data = []
    N = len(basis)

    # 1) calcola l’insieme degli indici e dei moltiplicatori (come in low4_index)
    for k_tuple, molt in rai1_low3_index(omega_list, lmax, Emax, M, R):

        k1, k2, k3, k4 = k_tuple
        # fattore scalare esterno
        factor = molt * (1.0 / (4 * math.sqrt(omega_list[k1] * omega_list[k2] * omega_list[k3] * omega_list[k4])))* (1.0 / (2 * math.pi * R))

     # 2) estrai le entry raw
        sub_rows, sub_cols, sub_vals = rai1_low3(basis, state_index, ix_list,  k1, k2, k3, k4)

    # 3) accumula, moltiplicando in blocco
        rows.extend(sub_rows)
        cols.extend(sub_cols)
        data.extend([factor * v for v in sub_vals])

    # 4) un’unica creazione COO + conversione a CSR
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

######################### phi-^2 phi+^2   ###########################
def rai2_low2_index(omega_list,lmax,Emax,M,R):
    list_index = []
    # -lmax <= k1 < k2 <= lmax
    # -lmax <= k2 < k3 <= lmax
    for k1 in range(-lmax, lmax):
        for k2 in range(k1+1, lmax+1):
            for k3 in range(-lmax, lmax):
                k4=k1+k2-k3
                if k4 in range(k3+1,lmax+1) and omega_list[k1]+ omega_list[k2]<= Emax and omega_list[k3]+ omega_list[k4] <= Emax:
                    index=[k1,k2,k3,k4]
                    m= 4 #2!*2!
                    list_index.append([index, m])

    # -lmax <= k1 < k2 <= lmax
    # -lmax <= k3 = k4 <= lmax
    for k1 in range(-lmax, lmax):
        for k2 in range(k1+1, lmax+1):
            if (k1+k2) % 2 == 0:
                k3 = int((k1+k2)/2)
                k4=k3
                if k4 in range(-lmax, lmax+1)  and omega_list[k1]+ omega_list[k2]<= Emax and omega_list[k3]+ omega_list[k4] <= Emax:
                    index=[k1,k2,k3,k4]
                    m= 2 #2!
                    list_index.append([index, m])

    # -lmax <= k1 = k2 <= lmax
    # -lmax <= k3 < k4 <= lmax
    for k3 in range(-lmax, lmax):
        for k4 in range(k3+1, lmax+1):
            if (k3+k4) % 2 == 0:
                k1 = int((k3+k4)/2)
                k2=k1
                if k1 in range(-lmax, lmax+1) and omega_list[k1]+ omega_list[k2]<= Emax and omega_list[k3]+ omega_list[k4] <= Emax:
                    index=[k1,k2,k3,k4]
                    m= 2 #2!
                    list_index.append([index, m])

    # -lmax <= k1 = k2 <= lmax
    # -lmax <= k3 = k3 <= lmax
    for k1 in range(-lmax, lmax+1):
        k2=k1
        k3=k1
        k4=k1
        if omega_list[k1]+ omega_list[k2]<= Emax and omega_list[k3]+ omega_list[k4] <= Emax:
            index=[k1,k2,k3,k4]
            m= 1 #2!/2!
            list_index.append([index, m])
    return list_index

def rai2_low2(basis,state_index, ix_list, k1,k2,k3,k4):
    # Computes matrix elements for a^2 (creation) and a^2 (annihilation) terms
    row=[]
    col=[]
    values=[]
    ix_k1= ix_list[k1]
    ix_k2= ix_list[k2]
    ix_k3= ix_list[k3]
    ix_k4= ix_list[k4]
    dim=len(basis)
    
    for state_ix in range(len(basis)):
        pos=None
        state = list(basis[state_ix])

        # Check n_k1 > 0
        if state[ix_k1] !=0: 
            factor1= math.sqrt(state[ix_k1])
            state[ix_k1] -= 1 
            # Check n_k2 > 0
            if state[ix_k2] !=0: 
                factor2= factor1*math.sqrt(state[ix_k2])
                state[ix_k2] -= 1

                factor3= factor2*math.sqrt(state[ix_k3]+1)
                state[ix_k3] += 1
                
                factor4= factor3*math.sqrt(state[ix_k4]+1)
                state[ix_k4] += 1
          
                pos = state_index.get(tuple(state))
                if pos != None:
                    row.append(pos)
                    col.append(state_ix)
                    values.append(factor4)

    return row, col, values

def Term_rai2_low2(basis,state_index, omega_list, ix_list, lmax, Emax, M, R):
    # Builds the phi-^2 phi+^2 term as a sparse matrix
    rows = []
    cols = []
    data = []
    N = len(basis)

    # 1) Compute index set and multiplicities
    for k_tuple, molt in rai2_low2_index(omega_list, lmax, Emax, M, R):

        k1, k2, k3, k4 = k_tuple
        # External scalar factor
        factor = molt * (1.0 / (4 * math.sqrt(omega_list[k1] * omega_list[k2] * omega_list[k3] * omega_list[k4])))* (1.0 / (2 * math.pi * R))

        # 2) Extract raw entries
        sub_rows, sub_cols, sub_vals = rai2_low2(basis, state_index, ix_list,  k1, k2, k3, k4)

        # 3) Accumulate, multiplying in bulk
        rows.extend(sub_rows)
        cols.extend(sub_cols)
        data.extend([factor * v for v in sub_vals])

    # 4) Single COO creation + conversion to CSR
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

def H4(basis, state_index, omega_list , ix_list, lmax, Emax, M, R):
    # Builds the full H4 matrix for the quartic interaction
    term_low4 = Term_low4(basis, state_index, omega_list,  ix_list, lmax, Emax, M, R)
    term_rai4 = term_low4.transpose().tocsr()
    term_rai1_low3 = Term_rai1_low3(basis, state_index, omega_list, ix_list, lmax, Emax, M, R)
    term_rai3_low1 = term_rai1_low3.transpose().tocsr()
    term_rai2_low2 = Term_rai2_low2(basis, state_index , omega_list, ix_list, lmax, Emax, M, R)
    H4 = term_rai4 + term_low4 + 4*term_rai1_low3 + 4*term_rai3_low1 + 6*term_rai2_low2
    return H4

##################################################################
################## Corrections O(V^2) ############################
##################################################################

#Correction to the coupling constant
def correction_lambda_VV(omega_list,Lambda,R,M,Emax,k_UV):
    prefactor = - (3*Lambda**2)/(16*math.pi*R)
    sum_term=0
    if 2*omega_list[0] >= Emax:
         sum_term += 1/(omega_list[0]**3)
    for k in range (1,k_UV +1): #sum over k>0 gives a factor 
        if 2*omega_list[k] >= Emax:
            sum_term += 2/(omega_list[k]**3)

    lambda_2 = prefactor*sum_term
    return lambda_2

#Correction to mass
def correction_m_VV(omega_list, Lambda,R,M,Emax,k_UV):
    prefactor1 = (Lambda)/(16*math.pi*R)
    prefactor2 = (Lambda)/(6*math.pi*R)
    sum_term=0

    #-K_UV <= k1 < k2 < k3 <= K_UV
    for k1 in range (-k_UV,k_UV-1):
        for k2 in range (k1+1,k_UV):
            k3=-k1-k2
            if k3 in range(k2+1,k_UV+1) and omega_list[k1] + omega_list[k2] + omega_list[k3]>=Emax:
                m=6 #3!
                sum_term += m*(1/((omega_list[k1] * omega_list[k2] * omega_list[k3])*(-omega_list[k1] - omega_list[k2] - omega_list[k3])))

    #-K_UV <= k1 = k2 < k3 <= K_UV
    for k1 in range (-k_UV,k_UV):
        k2=k1
        k3=-k1-k2 
        if k3 in range(k1+1,k_UV+1) and omega_list[k1] + omega_list[k2] + omega_list[k3]>=Emax:
            m=3 #3!/2!
            sum_term += m*(1/((omega_list[k1] * omega_list[k2] * omega_list[k3])*(-omega_list[k1] - omega_list[k2] - omega_list[k3])))

    #-K_UV <= k1 < k2 = k3 <= K_UV
    for k1 in range (-k_UV,k_UV):
        if k1 % 2 == 0:
            k3= - int(k1/2)
            k2=k3
            if k3 in range(k1+1,k_UV+1) and omega_list[k1] + omega_list[k2] + omega_list[k3]>=Emax:
                m=3 #3!/2!
                sum_term += m*(1/((omega_list[k1] * omega_list[k2] * omega_list[k3])*(-omega_list[k1] - omega_list[k2] - omega_list[k3])))

    #-K_UV <= k1 = k2 = k3 <= K_UV
    k1=0
    k2=0
    k3=0        
    if  omega_list[k1] + omega_list[k2] + omega_list[k3]>=Emax:
        m=1 #3!/3!!
        sum_term += m*(1/((omega_list[k1] * omega_list[k2] * omega_list[k3])*(-omega_list[k1] - omega_list[k2] - omega_list[k3])))

    m_2 = prefactor1*prefactor2*sum_term 
    return m_2


##################################################################
#########################  EIGENS ################################
##################################################################

def Eigens(H, N_eigens):
    dim_matrix = H.shape[0]

    if N_eigens > dim_matrix - 2:
        # Caso denso: usa eig
        eigvals, eigvecs = scipy.linalg.eig(H.toarray())
        eigvals = eigvals.real
        # Ordina e restituisce i primi N_eigens
        idx = np.argsort(eigvals)[:N_eigens]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    else:
        # Caso sparso: usa eigs
        H_norm = scipy.sparse.linalg.norm(H)
        eigvals, eigvecs = scipy.sparse.linalg.eigs(H - H_norm * scipy.sparse.identity(dim_matrix), 
                                                    k=N_eigens, 
                                                    which='LM')
        eigvals = eigvals.real + H_norm
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs

def Z_analysis_to_En(VecV, VecVV, basis, n_analysis=6, tol=1e-10):
    print(f"\n{'n':>4} | {'Z':^6}")
    print("-" * 15)
    for n in range(n_analysis):
        z_raw = Z_sign(n, VecV, basis, tol)
        z_imp = Z_sign(n, VecVV, basis,tol)
        
        if z_raw != z_imp:
            print(f"E_{n:<2} | Raw = {z_raw}, Improved = {z_imp}")
            continue
        print(f"E_{n:<2} | {z_raw:^6}")

def Z_sign(n, Vec,basis,tol=1e-10):
    Vec_En = Vec[:, n]
    z_signs = set()

    for j, coeff in enumerate(Vec_En):
        if np.abs(coeff) < tol:
            continue
        state = basis[j]
        ntot = n_total(state)
        z_signs.add(ntot % 2)
        
    if z_signs == {0}:
        return "+"
    elif z_signs == {1}:
        return "-"
    else:
        return "mix"

def check_violation_matrix(Heff_V, Heff_VV,basis):
    # Ciclo sugli elementi non nulli della matrice H
    rows, cols = Heff_V.nonzero()
    for i, j in zip(rows, cols):
        i=int(i)
        j=int(j)
        l_initial=l_total(basis[j])
        l_final=l_total(basis[i])
        if l_initial != l_final:
            print("\nError: momento non conservato!")
            exit(1)

    rows, cols = Heff_VV.nonzero()
    for i, j in zip(rows, cols):
        i=int(i)
        j=int(j)
        l_initial=l_total(basis[j])
        l_final=l_total(basis[i])
        if l_initial != l_final:
            print("\nError: momento non conservato!")
            exit(1)

def p_analysis_to_En(VecV, VecVV, basis, n_analysis=6, tol=1e-10):
    print(f"\n{'n':>4} | {'Momentum':^9}")
    print("-" * 15)
    for n in range(n_analysis):
        p_raw = p_value(n, VecV, basis, tol)
        p_imp = p_value(n, VecVV,  basis, tol)

        if p_raw != p_imp:
            print(f"E_{n:<2} | Raw = {p_raw} , Improved = {p_imp}")
            continue
            
        print(f"E_{n:<2} | {p_raw:^9}")

def p_value(n, Vec, basis, tol=1e-10):
    vec_n = Vec[:, n]
    P_values = set()
    for j, coeff in enumerate(vec_n):
        if np.abs(coeff) < tol:
            continue
        state = basis[j]
        P_values.add(l_total(state))
    
    if len(P_values) == 1:
        # estrae l'unico elemento dall'insieme
        return P_values.pop()
    else:
        return "mix"

def Analysis_Eigensvector_n(n, basis, Vec, tol=1e-10):
    #Estrai la colonna dell'autovettore
    vec_n = Vec[:, n].copy()
    
    #Possibili errori
    dim, m = Vec.shape
    if not (0 <= n < m):
        raise IndexError(f"Indice dell'autovettore n={n} fuori dall'intervallo [0, {m-1}].")
    if len(basis) != dim:
        raise ValueError(f"Lunghezza della base ({len(basis)}) non coincide con la dimensione degli autovettori ({dim}).")
    if np.any(np.imag(vec_n) > 1e-12):
        raise ValueError("Attenzione: l'autovettore ha parte immaginaria non trascurabile.")
    

    #Salva c_i e stato
    coeff_list = []
    state_list = []
    
    for i, c_i in enumerate(vec_n):
            if np.abs(c_i)>tol:
                coeff_list.append(c_i.real)
                state_list.append(basis[i])

    norm = np.linalg.norm(coeff_list)
    coeff_list_normalized = [c / norm for c in coeff_list]
    if np.linalg.norm(coeff_list_normalized) < 1-tol:
        raise ValueError(f"Normalizzazione coefficienti rotta: {np.linalg.norm(coeff_list_normalized)}")
    
    return coeff_list_normalized, state_list

def Eigenvector_n(basis, H4_matrix, n, M, R, Lambda):
  
    E_0_list = [state_energy(state, M, R) for state in basis]
    E_n0 = E_0_list[n]
    vec_n0 = basis[n]

    # Inizializza |E_n> = |E_n^0> + ...
    coeffs = [1.0]               # coefficiente del vettore non perturbato
    states = [vec_n0]            # stato non perturbato

    # Converti in formato CSC per accesso efficiente alla colonna n
    H4_csc = H4_matrix.tocsc()
    start_ptr = H4_csc.indptr[n]
    end_ptr = H4_csc.indptr[n + 1]
    rows = H4_csc.indices[start_ptr:end_ptr]
    values = H4_csc.data[start_ptr:end_ptr]

    for m, V_mn in zip(rows, values):
        if m == n:
            continue
        E_m0 = E_0_list[m]
        denom = E_n0 - E_m0
        if abs(denom) < 1e-14:
            # Evita instabilità numeriche
            continue
        corrected_coeff = (Lambda / 24.0) * V_mn / denom
        coeffs.append(corrected_coeff)
        states.append(basis[m])
    
    norm_squared = sum(abs(c)**2 for c in coeffs)
    norm = norm_squared**0.5

    # Normalizza i coefficienti
    coeffs = [c / norm for c in coeffs]
    return coeffs, states


##################################################################
#########################  SAVING  ###############################
##################################################################
def moments_to_filename(allowed_l):
    allowed_l_str = "_".join(f"l{abs(l)}" if l >= 0 else f"lm{abs(l)}" for l in allowed_l)
    return allowed_l_str

def save_eigenvectors(moments, R, M, Lambda, basis, VecVV, Emax, mode, N_eigens):
    for n in range(N_eigens):
        coeffs, states = Analysis_Eigensvector_n(n,basis,VecVV)

        # 1) Percorso del database e caricamento (o inizializzazione)
        db_path = "Data/database.yaml"
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                config_db = yaml.safe_load(f) or {}
        else:
            config_db = {}

        # 2) Definizione della configurazione corrente
        current_cfg = {"Lambda": Lambda, "M": M, "R": R}

        # 3) Ricerca di una voce esistente identica
        cfg_name = None
        for name, cfg in config_db.items():
            if cfg == current_cfg:
                cfg_name = name
                break

        # 4) Creazione di una nuova voce in caso non esista
        if cfg_name is None:
            cfg_name = f"config{len(config_db) + 1}"
            config_db[cfg_name] = current_cfg
            os.makedirs("Data", exist_ok=True)
            with open(db_path, 'w') as f:
                yaml.dump(config_db, f)
            print(f"Added new configuration → {cfg_name}")


        # 5) Creazione del percorso per i momenti e gli autovettori
        moments_str = moments_to_filename(moments)
        base = os.path.join("Data", cfg_name, f"Moments_{moments_str}", "Eigenvectors")
        if mode == 'even':
            folder = base + "_even"
        elif mode == 'odd':
            folder = base + "_odd"
        else:
            folder = base
        os.makedirs(folder, exist_ok=True)

        # 6) Salvataggio dell'n-esimo autovettore
        #    Usiamo .npz per mantenere array multipli (coeffs, states) in un unico file
        filename = f"Eigenvec_Emax{Emax}_n{n}.npz"
        path = os.path.join(folder, filename)
        np.savez(path,
                coeffs = np.asarray(coeffs),
                states = np.asarray(states))

    print(f"Saved eigenvector #{N_eigens}")

def save_eigenvalues(moments, R, M, Lambda, EigV, EigVV, Emax, mode, N_eigens):
    # 1) Carica il database delle configurazioni
    database_path = "Data/database.yaml"
    if os.path.exists(database_path):
        with open(database_path, 'r') as f:
            config_db = yaml.safe_load(f) or {}
    else:
        config_db = {}

    # 2) Prepara il dizionario della nuova configurazione
    new_config = {"Lambda": Lambda, "M": M, "R": R}

    # 3) Cerca se esiste già una configurazione identica
    config_name = None
    for name, cfg in config_db.items():
        if cfg == new_config:
            config_name = name
            break

    # 4) Se non esiste, crea una nuova entry
    if config_name is None:
        config_name = f"config{len(config_db)+1}"
        config_db[config_name] = new_config
        # salva il nuovo database aggiornato
        os.makedirs("Data", exist_ok=True)
        with open(database_path, 'w') as f:
            yaml.dump(config_db, f)
        print(f"Added new configuration as {config_name} in database.")
    else:
        print(f"Configuration already present as {config_name}.")
        
    # 5) Crea il percorso specifico per la configurazione e i momenti
    moments_str = moments_to_filename(moments)
    base_folder = os.path.join("Data", config_name, f"Moments_{moments_str}", "Eigenvalues")

    if mode == 'even':
        folder = base_folder + "_even"
    elif mode == 'odd':
        folder = base_folder + "_odd"
    else:
        folder = base_folder

    os.makedirs(folder, exist_ok=True)

    # 6) Costruisce i nomi dei file per gli autovalori
    path_V  = os.path.join(folder, f"Eigen_Emax{Emax}_V.txt")
    path_VV = os.path.join(folder, f"Eigen_Emax{Emax}_VV.txt")

    # 7) Salva i primi N_eigens autovalori
    np.savetxt(path_V,  EigV[:N_eigens])
    np.savetxt(path_VV, EigVV[:N_eigens])

    print(f"Saved eigenvalues under {folder}")

def save_correction(moments, R, M, Lambda, m_2, Lambda_2, Emax, mode):
       # 1) Carica il database delle configurazioni
    database_path = "Data/database.yaml"
    if os.path.exists(database_path):
        with open(database_path, 'r') as f:
            config_db = yaml.safe_load(f) or {}
    else:
        config_db = {}

    # 2) Prepara il dizionario della nuova configurazione
    new_config = {"Lambda": Lambda, "M": M, "R": R}

    # 3) Cerca se esiste già una configurazione identica
    config_name = None
    for name, cfg in config_db.items():
        if cfg == new_config:
            config_name = name
            break

    # 4) Se non esiste, crea una nuova entry
    if config_name is None:
        config_name = f"config{len(config_db)+1}"
        config_db[config_name] = new_config
        # salva il nuovo database aggiornato
        os.makedirs("Data", exist_ok=True)
        with open(database_path, 'w') as f:
            yaml.dump(config_db, f)
        print(f"Added new configuration as {config_name} in database.")
    else:
        print(f"Configuration already present as {config_name}.")
        
    # 5) Crea il percorso specifico per la configurazione e i momenti
    moments_str = moments_to_filename(moments)
    folder = os.path.join("Data", config_name, f"Moments_{moments_str}", "Corrections")

    os.makedirs(folder, exist_ok=True)

    # 6) Costruisce i nomi dei file per gli autovalori
    path = os.path.join(folder, f"Corr_Emax{Emax}.txt")

    # 7) Salva i primi N_eigens autovalori
    np.savetxt(path, [[m_2, Lambda_2]])
    

    print(f"Saved Correction under {folder}")

def save_time_number(moments, R, M, Lambda, Emax, number_states, time_taken):
    """
    Save computation cost (number of states and time used) for a given configuration and moments.
    Stores results in Data/database.yaml for configurations and in a per-configuration file:
    Data/configX/Moments_<moments>/Computation/computation_cost.txt
    The computation_cost.txt file has a header and entries for unique Emax values only.
    """
    database_path = os.path.join("Data", "database.yaml")
    # Load or initialize configuration database
    if os.path.exists(database_path):
        with open(database_path, 'r') as f:
            config_db = yaml.safe_load(f) or {}
    else:
        config_db = {}

    # Prepare the new configuration dict
    new_config = {"Lambda": Lambda, "M": M, "R": R}

    # Check if config exists
    config_name = None
    for name, cfg in config_db.items():
        if cfg == new_config:
            config_name = name
            break

    # Add new config if needed
    if config_name is None:
        config_name = f"config{len(config_db) + 1}"
        config_db[config_name] = new_config
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        with open(database_path, 'w') as f:
            yaml.dump(config_db, f)
        print(f"Added new configuration as {config_name} in database.")
    else:
        print(f"Configuration already present as {config_name}.")

    # Prepare folder path
    moments_str = moments_to_filename(moments)
    comp_dir = os.path.join("Data", config_name, f"Moments_{moments_str}", "Computation")
    os.makedirs(comp_dir, exist_ok=True)

    # Path for computation costs file
    cost_file = os.path.join(comp_dir, "computation_cost.txt")

    # Header and new entry
    header = "Emax Nstate TimeUsed_secondes"
    new_entry = f"{Emax} {number_states} {time_taken}"

    # If file does not exist, create with header
    if not os.path.exists(cost_file):
        with open(cost_file, 'w') as f:
            f.write(header + "\n")
            f.write(new_entry + "\n")
        print(f"Initialized computation_cost.txt with Emax={Emax}.")
    else:
        # Read existing entries
        with open(cost_file, 'r') as f:
            lines = f.readlines()
        # Parse existing Emax values
        existing = {line.split()[0] for line in lines[1:] if line.strip()}
        # Append only if new Emax not present
        if str(Emax) not in existing:
            with open(cost_file, 'a') as f:
                f.write(new_entry + "\n")
            print(f"Appended Emax={Emax} to computation_cost.txt.")
        else:
            print(f"Emax={Emax} already recorded; no update made.")

##################################################################
############################# MAIN ###############################
##################################################################

def Basis(Emax, moments, R, M, mode):
    print(f"Momento: {moments} \t Emax: {Emax} \t R: {R} \t Mode: {mode}")

    # Generazione della base
    N = 4.0
    if M**2 < (Emax**2) / N:
        lmax = int(math.floor(math.sqrt(R**2 * (Emax**2 / N - M**2))))
    else:
        lmax = 0
    # Creazione di omega_list
    omega_list = {k: omega(k, M, R) for k in range(-lmax, lmax + 1)}
    start = time.time()
    if mode == 'even':
        basis = sorted(filter_even(filter_moments(moments, gen_basis(omega_list, lmax, Emax, M, R))),
                      key=lambda state: (abs(l_total(state)), l_total(state) < 0, state_energy(state, M, R)))
    elif mode == 'odd':
        basis = sorted(filter_odd(filter_moments(moments, gen_basis(omega_list, lmax, Emax, M, R))),
                      key=lambda state: (abs(l_total(state)), l_total(state) < 0, state_energy(state, M, R)))
    else:
        basis = sorted(filter_moments(moments, gen_basis(omega_list, lmax, Emax, M, R)),
                      key=lambda state: (abs(l_total(state)), l_total(state) < 0, state_energy(state, M, R)))
    end = time.time()
    number_states = len(basis)
    time_taken = end - start
    return basis, number_states, time_taken

def Matrices(Emax, basis,R,M,Lambda,N_eigens):
    # Definizione lmax
    N = 4.0
    if M**2 < (Emax**2) / N:
        lmax = int(math.floor(math.sqrt(R**2 * (Emax**2 / N - M**2))))
    else:
        lmax = 0
    # Creazione di omega_list
    omega_list = {k: omega(k, M, R) for k in range(-lmax, lmax + 1)}

    # Creazione indici
    state_index = {tuple(s): i for i, s in enumerate(basis)}
    ix_list = {l: (2 * abs(l) if l < 0 else (2 * l - 1) if l > 0 else 0) for l in range(-lmax, lmax + 1)}

    # Costruzione matrici di Heff
    H0_matrix = H0(basis, M, R)
    H2_matrix = H2(basis, state_index, omega_list, ix_list, lmax, Emax, M, R)
    H4_matrix = H4(basis, state_index, omega_list, ix_list, lmax, Emax, M, R)
    Heff_V = H0_matrix + (Lambda / 24) * H4_matrix

    # Correzioni O(VV)
    k_UV = 1000
    omega_list_UV = {k: omega(k, M, R) for k in range(-k_UV, k_UV + 1)}
    m_2 = correction_m_VV(omega_list_UV, Lambda, R, M, Emax, k_UV)
    Lambda_2 = correction_lambda_VV(omega_list_UV, Lambda, R, M, Emax, k_UV)
    Heff_VV = ( H0_matrix + (m_2 / 2) * H2_matrix + ((Lambda + Lambda_2) / 24) * H4_matrix)
    check_violation_matrix(Heff_V, Heff_VV,basis)
    print("\n✔️ Matrici generate e Momento Conservato.")
    
    
    
    # Calcolo degli autovalori e Testing
    EigV, VecV  = Eigens(Heff_V,  N_eigens)
    EigVV, VecVV = Eigens(Heff_VV, N_eigens)

    print("\n✔️ Autovalori e Vettori Estratti.")

    """
    n=1
    vec_pert = build_perturbative_vector(basis, H4_matrix, n, M, R, Lambda)
    vec_num = VecV[:, n]
    vec_num /= np.linalg.norm(vec_num)

    fidelity, angle = compare_vectors(vec_pert, vec_num)
    print(f"\nFidelity: {fidelity:.6f} — Angle (rad): {angle:.3e}")
    """
    return Heff_V, Heff_VV, m_2, Lambda_2, EigV, VecV, EigVV, VecVV


def build_perturbative_vector(basis, H4_matrix, n, M, R, Lambda):
    coeffs, states = Eigenvector_n(basis, H4_matrix, n, M, R, Lambda)

    vec = np.zeros(len(basis))
    for coeff, state in zip(coeffs, states):
        index = basis.index(state)  # posizione dello stato nella base
        vec[index] = coeff

    # Normalizza
    vec /= np.linalg.norm(vec)

    return vec

def compare_vectors(v1, v2, tol=1e-12):
    overlap = np.vdot(v1, v2)
    fidelity = abs(overlap)**2  # tra 0 e 1
    angle = np.arccos(min(1.0, max(-1.0, np.real(overlap))))
    return fidelity, angle

if __name__ == '__main__':
# =============================================================
# Main script for Hamiltonian truncation and eigenvalue analysis
# Author: Andrea Maestri, University of Pavia
# Some code portions adapted from https://github.com/rahoutz/hamiltonian-truncation
# This script generates the basis, builds effective Hamiltonians, computes eigenvalues/vectors,
# saves results, and performs several physical analyses including perturbative theory checks.
# =============================================================
    
    #Parsing degli argomenti
    parser = argparse.ArgumentParser(description="Calcolo degli autovalori con opzione even/odd basis")
    parser.add_argument('Emax',type=int,help="Massima energia Emax")
    parser.add_argument('Moments',type=str,help="Lista degli indici l consentiti (es. '0,1')")
    parser.add_argument('Ray',type=float,help="Fattore Raggio")
    parser.add_argument('La',type=float,help="Fattore Lambda")
    parser.add_argument('mode',nargs='?',choices=['even', 'odd'],default=None,help="Modalità di selezione della base: 'even' o 'odd'")
    args = parser.parse_args()

    #Argomenti da Parsing
    
    Moments = list(map(int, args.Moments.split(',')))
    if len(Moments) != 1:
        print("Errore: è necessario fornire esattamente un solo momento.")
        exit(1)
    else:
        N_tot = Moments[0]
    
    mode = args.mode
    Ray = args.Ray
    La = args.La
    Lambda = 4 * math.pi * La
    N_eigens = 6

    R = (10.0 / (2 * math.pi)) * Ray
    #R =  Ray
    M = 1.0
    if N_tot == 0:
        Emax = args.Emax
    else:
        Emax = np.sqrt((args.Emax)**2+(N_tot/R)**2)

    #Generazione base, matrici, autovalori a autovettori.
    basis, number_states, time_taken = Basis(Emax, Moments, R, M, mode)
    
    Heff_V, Heff_VV, m_2, Lambda_2 , EigV, VecV, EigVV, VecVV = Matrices(Emax,basis,R,M,Lambda,N_eigens) 

    save_time_number(Moments, R, M, Lambda, Emax, number_states, time_taken)
    save_correction(Moments, R, M, Lambda, m_2, Lambda_2, args.Emax, mode)
    save_eigenvalues(Moments, R, M, Lambda, EigV, EigVV, args.Emax, mode, N_eigens)
    save_eigenvectors(Moments, R, M, Lambda, basis, VecVV, args.Emax, mode, N_eigens)

    #Z_analysis_to_En(VecV, VecVV, basis, n_analysis=6, tol=1e-10)
    #p_analysis_to_En(VecV, VecVV, basis, n_analysis=6, tol=1e-10)


