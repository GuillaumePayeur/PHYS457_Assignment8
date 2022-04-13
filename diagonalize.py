import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def state_to_index(state,base):
    index = int(state,base)
    return index

def index_to_state(index,base):
    if index == 0:
        return '000000'
    state = ''
    while index:
        state += str(index % base)
        index //= base
    while len(state) < 6:
        state += '0'
    return state[::-1]

def a(n,state,base):
    if int(state[n]) > 0:
        resulting_state = state[:n] + str(int(state[n])-1) + state[n+1:]
        return (1/np.sqrt(int(state[n])),resulting_state)
    return None

def a_dagger(n,state,base):
    if int(state[n]) < (base-1):
        resulting_state = state[:n] + str(int(state[n])+1) + state[n+1:]
        return (1/np.sqrt(int(state[n])+1),resulting_state)
    return None

def check_energy_cutoff(state,base):
    sum_n = 0
    for i in range(6):
        sum_n += int(state[i])
    if sum_n <= (base-1):
        return True
    return False

def sum_n(state):
    sum_n = 0
    for i in range(6):
        sum_n += int(state[i])
    return sum_n

def cut_Hamiltonian(H,base,n=0):
    indices_to_remove = []
    for index in range(base**6):
        if not check_energy_cutoff(index_to_state(index,base),base):
            indices_to_remove.append(index)
    H = np.delete(H,np.array(indices_to_remove),axis=0)
    H = np.delete(H,np.array(indices_to_remove),axis=1)
    return H

def cut_Hamiltonian_restricted(H,base,n=0):
    indices_to_remove = []
    for index in range(base**6):
        if sum_n(index_to_state(index,base)) != n:
            indices_to_remove.append(index)

    H = np.delete(H,np.array(indices_to_remove),axis=0)
    H = np.delete(H,np.array(indices_to_remove),axis=1)
    return H

def get_perturbing_H(H_terms_pos,H_terms_neg,base,cutting_fn,n=0):
    H1 = np.zeros((base**6,base**6))
    for index in range(base**6):
        if check_energy_cutoff(index_to_state(index,base),base):
            for term in H_terms_pos:
                state = index_to_state(index,base)

                state1 = a(term[0],state,base)
                if state1 != None:
                    state2 = a(term[1],state1[1],base)
                    if state2 != None:
                        H1[index,state_to_index(state2[1],base)] += state1[0]*state2[0]

                state1 = a(term[0],state,base)
                if state1 != None:
                    state2 = a_dagger(term[1],state1[1],base)
                    if state2 != None:
                        if check_energy_cutoff(state2[1],base):
                            H1[index,state_to_index(state2[1],base)] += state1[0]*state2[0]

                H1[index,state_to_index(state,base)] += 1/(int(state[term[0]])+1)

                state1 = a_dagger(term[0],state,base)
                if state1 != None:
                    state2 = a_dagger(term[1],state1[1],base)
                    if state2 != None:
                        H1[index,state_to_index(state2[1],base)] += state1[0]*state2[0]

            for term in H_terms_neg:
                state = index_to_state(index,base)

                state1 = a(term[0],state,base)
                if state1 != None:
                    state2 = a(term[1],state1[1],base)
                    if state2 != None:
                        H1[index,state_to_index(state2[1],base)] -= state1[0]*state2[0]

                state1 = a(term[0],state,base)
                if state1 != None:
                    state2 = a_dagger(term[1],state1[1],base)
                    if state2 != None:
                        H1[index,state_to_index(state2[1],base)] -= state1[0]*state2[0]

                state1 = a_dagger(term[0],state,base)
                if state1 != None:
                    state2 = a(term[1],state1[1],base)
                    if state2 != None:
                        H1[index,state_to_index(state2[1],base)] -= state1[0]*state2[0]

                state1 = a_dagger(term[0],state,base)
                if state1 != None:
                    state2 = a_dagger(term[1],state1[1],base)
                    if state2 != None:
                        H1[index,state_to_index(state2[1],base)] -= state1[0]*state2[0]
    H1 = cutting_fn(H1,base,n)
    return H1

def get_noninteracting_H(base):
    H0 = np.zeros((base**6,base**6))
    for index in range(base**6):
        state = index_to_state(index,base)
        if check_energy_cutoff(state,base):
            H0[index,index] = (sum_n(state)+3)
    H0 = cut_Hamiltonian(H0,base)
    return H0

def diagonalize(H):
    return np.linalg.eig(H)[0]

def plot_spectrum_varying_w(base,wi,wf,n,x):
    H_terms_pos = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
    H_terms_neg = [(0,3),(1,4),(2,5),(3,0),(4,1),(5,2)]

    spectrum_array = []
    for w in np.linspace(wi,wf,n):
        w0 = x*w
        H0 = hbar*w*get_noninteracting_H(base)
        H1 = hbar*w0**2/(w*4)*get_perturbing_H(H_terms_pos,H_terms_neg,base,cut_Hamiltonian)
        H = H0 + H1

        spectrum = diagonalize(H)
        spectrum_array.append(np.sort(spectrum))

    spectrum_array = np.round(np.real(np.array(spectrum_array)),4)
    unique = list(np.unique(spectrum_array[1,:],return_index=True)[1])
    for i in range(spectrum_array.shape[1]):
        plt.plot(np.linspace(wi,wf,n),spectrum_array[:,i],color='green',lw=0.5)
    plt.xlabel('$\omega$')
    plt.ylabel('E')
    plt.show()

def plot_spectrum_varying_w0(base,w0i,w0f,n):
    w = 1
    H_terms_pos = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
    H_terms_neg = [(0,3),(1,4),(2,5),(3,0),(4,1),(5,2)]

    spectrum_array = []
    for w0 in np.linspace(w0i,w0f,n):
        H0 = hbar*w*get_noninteracting_H(base)
        H1 = hbar*w0**2/(w*4)*get_perturbing_H(H_terms_pos,H_terms_neg,base,cut_Hamiltonian)
        H = H0 + H1

        spectrum = diagonalize(H)
        spectrum_array.append(np.sort(spectrum))

    spectrum_array = np.round(np.real(np.array(spectrum_array)),4)
    unique = list(np.unique(spectrum_array[1,:],return_index=True)[1])
    for i in range(spectrum_array.shape[1]):
        plt.plot(np.linspace(w0i,w0f,n),spectrum_array[:,i],color='green',lw=0.5)
    plt.xlabel('$\omega_0$')
    plt.ylabel('E')
    plt.show()

def get_H0(base):
    w = 1
    H_terms_pos = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
    H_terms_neg = [(0,3),(1,4),(2,5),(3,0),(4,1),(5,2)]

    H0 = w*get_noninteracting_H(base)
    return H0

def get_H1(base,w0,cutting_fn,n=0):
    w = 1
    H_terms_pos = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
    H_terms_neg = [(0,3),(1,4),(2,5),(3,0),(4,1),(5,2)]

    H1 = hbar*w0**2/(w*4)*get_perturbing_H(H_terms_pos,H_terms_neg,base,cutting_fn,n)
    return H1

def plot_energy_corrections(base,w0i,w0f,n,energy_level):
    w = 1
    H_terms_pos = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
    H_terms_neg = [(0,3),(1,4),(2,5),(3,0),(4,1),(5,2)]

    spectrum_array = []
    for w0 in np.linspace(w0i,w0f,n):
        H1 = hbar*w0**2/(w*4)*get_perturbing_H(H_terms_pos,H_terms_neg,base,cut_Hamiltonian_restricted,energy_level)

        spectrum = diagonalize(H1)
        spectrum_array.append(np.sort(spectrum))

    spectrum_array = np.real(np.array(spectrum_array))
    for i in range(spectrum_array.shape[1]):
        plt.plot(np.linspace(w0i,w0f,n),spectrum_array[:,i]+3+energy_level,color='green',lw=0.5)
    plt.xlabel('$\omega_0/\omega$')
    plt.ylabel('E')
    plt.show()

if __name__ == '__main__':
    hbar = 1

    print(get_H0(2))
    print(get_H1(2,0.1,cut_Hamiltonian))
    plot_spectrum_varying_w(4,0.01,2,21,0)
    plot_spectrum_varying_w(4,0.01,2,21,0.3)
    plot_spectrum_varying_w0(4,0,2,501)

    print(np.round(get_H1(2,1,cut_Hamiltonian_restricted,n=0),3))
    print(np.round(get_H1(2,1,cut_Hamiltonian_restricted,n=1),3))
    plot_energy_corrections(4,0,1,100,0)
    plot_energy_corrections(4,0,1,100,1)
