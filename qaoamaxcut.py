from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import numpy as np

def qaoa_maxcut_K4_pn(p, shots=1024):
    # K4 edges
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    n_qubits = 4
    N = 2**n_qubits

    # Precompute Z_i Z_j masks
    Z_masks = []
    for (i, j) in edges:
        mask = np.array([1 if ((k >> i) & 1) == ((k >> j) & 1) else -1
                         for k in range(N)])
        Z_masks.append(mask)
    Z_masks = np.array(Z_masks)

    # Hamiltonian expectation
    def hamiltonian_expectation(statevec):
        prob = np.abs(statevec)**2
        zz_vals = Z_masks @ prob
        return 0.5 * np.sum(1 - zz_vals)

    # QAOA circuit for general p
    def build_circuit(params):
        gamma = params[:p]
        beta = params[p:]
        qc = QuantumCircuit(n_qubits)

        # Initial |+>
        for q in range(n_qubits):
            qc.h(q)

        # p layers
        for layer in range(p):
            # Cost unitary
            g = gamma[layer]
            for i, j in edges:
                qc.cx(i, j)
                qc.rz(2 * g, j)
                qc.cx(i, j)

            # Mixer
            b = beta[layer]
            for q in range(n_qubits):
                qc.rx(2 * b, q)

        return qc

    # Cost function
    def cost_fn(params):
        qc = build_circuit(params)
        state = Statevector.from_instruction(qc).data
        return -hamiltonian_expectation(state)

    # Initial guess: all parameters = 0.50
    init_guess = np.array([0.50] * (2 * p))
    bounds = [(0, np.pi)] * (2 * p)

    # Classical optimization
    res = minimize(cost_fn, x0=init_guess, bounds=bounds,
                   method='COBYLA', options={'maxiter': 300 + 50*p})

    gamma_opt = res.x[:p]
    beta_opt  = res.x[p:]

    # Final statevector
    qc_final = build_circuit(res.x)
    state = Statevector.from_instruction(qc_final).data
    probs = np.abs(state)**2

    # Counts
    bitstrings = [format(i, '04b') for i in range(N)]
    counts = {bs: int(probs[i] * shots) for i, bs in enumerate(bitstrings)}

    # Cut value function
    def cut_value(bs):
        return sum(1 for (i, j) in edges if bs[i] != bs[j])

    max_cut = max(cut_value(bs) for bs in bitstrings)

    # Print
    print(f"Optimized gamma: {gamma_opt}")
    print(f"Optimized beta: {beta_opt}")
    print(f"Maximum cut value found: {max_cut}")
    print(f"Counts summary out of {shots} shots:")
    for count_value in sorted(set(counts.values()), reverse=True):
        group = [bs for bs,c in counts.items() if c == count_value]
        print(f"{count_value} counts: {group}")

    return {'gamma': gamma_opt, 'beta': beta_opt}, counts, max_cut

# Replace p with any number greater than 0
if __name__ == "__main__":
    qaoa_maxcut_K4_pn(p=1)
