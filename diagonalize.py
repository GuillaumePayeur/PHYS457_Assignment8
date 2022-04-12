import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})
import matplotlib as mpl

def state_to_index(state):
    index = int(state,4)
    return index

def index_to_state(index):
    if index == 0:
        return '000000'
    state = ''
    while index:
        state += str(index % 4)
        index //= 4
    while len(state) < 6:
        state += '0'
    return state[::-1]

def a(n,state):
    if int(state[n]) > 0:
        resulting_state = state[:n] + str(int(state[n])-1) + state[n+1:]
        return (1/np.sqrt(int(state[n])),resulting_state)
    return None

def a_dagger(n,state):
    if int(state[n]) < 3:
        resulting_state = state[:n] + str(int(state[n])+1) + state[n+1:]
        return (1/np.sqrt(int(state[n])+1),resulting_state)
    return None

def check_energy_cutoff(state):
    sum_n = 0
    for i in range(6):
        sum_n += int(state[i])
    if sum_n <= 3:
        return True
    return False

def sum_n(state):
    sum_n = 0
    for i in range(6):
        sum_n += int(state[i])
    return sum_n

def cut_Hamiltonian(H):
    indices_to_remove = []
    for index in range(4096):
        if not check_energy_cutoff(index_to_state(index)):
            indices_to_remove.append(index)
    H = np.delete(H,np.array(indices_to_remove),axis=0)
    H = np.delete(H,np.array(indices_to_remove),axis=1)
    return H

def get_perturbing_H(H_terms):
    H1 = np.zeros((4096,4096))
    for index in range(4096):
        for term in H_terms:
            state = index_to_state(index)

            state1 = a(term[0],state)
            if state1 != None:
                state2 = a(term[1],state1[1])
                if state2 != None:
                    if check_energy_cutoff(state2[1]):
                        H1[index,state_to_index(state2[1])] += state1[0]*state2[0]

            state1 = a(term[0],state)
            if state1 != None:
                state2 = a_dagger(term[1],state1[1])
                if state2 != None:
                    if check_energy_cutoff(state2[1]):
                        H1[index,state_to_index(state2[1])] += state1[0]*state2[0]

            state1 = a_dagger(term[0],state)
            if state1 != None:
                state2 = a(term[1],state1[1])
                if state2 != None:
                    if check_energy_cutoff(state2[1]):
                        H1[index,state_to_index(state2[1])] += state1[0]*state2[0]

            state1 = a_dagger(term[0],state)
            if state1 != None:
                state2 = a_dagger(term[1],state1[1])
                if state2 != None:
                    if check_energy_cutoff(state2[1]):
                        H1[index,state_to_index(state2[1])] += state1[0]*state2[0]
    H1 = cut_Hamiltonian(H1)
    return H1

def get_noninteracting_H():
    H0 = np.zeros((4096,4096))
    for index in range(4096):
        state = index_to_state(index)
        if check_energy_cutoff(state):
            H0[index,index] = (sum_n(state)+3)
    H0 = cut_Hamiltonian(H0)
    return H0

def diagonalize(H):
    return np.linalg.eig(H)[0]

def plot_spectrum_fixed_w0(w0):
    w = 1
    H_terms = [(0,3),(1,4),(2,5)]

    H0 = np.sqrt(w**2+w0**2)*get_noninteracting_H()
    H1 = w0/2*get_perturbing_H(H_terms)
    H = H0 + H1

    spectrum = diagonalize(H)

    for eigenvalue in spectrum:
        plt.plot([0,1],[np.real(eigenvalue),np.real(eigenvalue)])
    plt.show()

def plot_spectrum_varying_w0(w0i,w0f,n):
    w = 1
    H_terms = [(0,3),(1,4),(2,5)]

    spectrum_array = []
    for w0 in np.linspace(w0i,w0f,n):
        H0 = np.sqrt(w**2+w0**2)*get_noninteracting_H()
        H1 = w0/2*get_perturbing_H(H_terms)
        H = H0 + H1

        spectrum = diagonalize(H)
        spectrum_array.append(np.sort(spectrum))

    spectrum_array = np.real(np.array(spectrum_array))
    for i in range(84):
        plt.plot(np.linspace(w0i,w0f,n),spectrum_array[:,i],color='black',lw=0.5)
    plt.xlabel('$\omega_0/\omega$')
    plt.ylabel('E')
    plt.savefig('test.png',dpi=500)
    plt.show()

if __name__ == '__main__':
    plot_spectrum_varying_w0(0,1,30)

# state = '200000'
# index = state_to_index(state)
# print(index)
# print(a(0,index))
# print(index_to_state(a(0,index)[1]))

# index = state_to_index('001004')
# print(check_energy_cutoff(index))
# quit()
