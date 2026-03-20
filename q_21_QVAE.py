"""
QVAE - Quantum Variational Autoencoder
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
LATENT_QUBITS = 2
TOTAL_QUBITS = NUM_QUBITS + LATENT_QUBITS
NUM_LAYERS = 2
MAXITER = 250
BETA_KL = 0.1


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def encoder_circuit(x, theta_enc, n_data, n_latent):
    n_total = n_data + n_latent
    qc = QuantumCircuit(n_total)

    for i in range(n_data):
        qc.ry(x[i], i)

    idx = 0
    for i in range(n_total):
        qc.ry(theta_enc[idx], i)
        idx += 1
    for i in range(n_total - 1):
        qc.cx(i, i + 1)
    for i in range(n_total):
        qc.rz(theta_enc[idx], i)
        idx += 1

    return qc, idx


def decoder_circuit(theta_dec, n_data, n_latent):
    n_total = n_data + n_latent
    qc = QuantumCircuit(n_total)

    idx = 0
    for layer in range(NUM_LAYERS):
        for i in range(n_total):
            qc.ry(theta_dec[idx], i)
            idx += 1
        for i in range(n_total - 1):
            qc.cx(i, i + 1)
        qc.cx(n_total - 1, 0)
        for i in range(n_total):
            qc.rz(theta_dec[idx], i)
            idx += 1

    return qc, idx


def num_encoder_params():
    return TOTAL_QUBITS * 2


def num_decoder_params():
    return NUM_LAYERS * TOTAL_QUBITS * 2


def forward(x, theta):
    n_enc = num_encoder_params()
    theta_enc = theta[:n_enc]
    theta_dec = theta[n_enc:]

    qc = QuantumCircuit(TOTAL_QUBITS)

    enc, _ = encoder_circuit(x, theta_enc, NUM_QUBITS, LATENT_QUBITS)
    qc.compose(enc, inplace=True)

    dec, _ = decoder_circuit(theta_dec, NUM_QUBITS, LATENT_QUBITS)
    qc.compose(dec, inplace=True)

    sv = Statevector.from_instruction(qc)
    full_probs = sv.probabilities()

    n_data_states = 1 << NUM_QUBITS
    marginal = np.zeros(n_data_states)
    n_latent_states = 1 << LATENT_QUBITS
    for i, p in enumerate(full_probs):
        data_bits = i % n_data_states
        marginal[data_bits] += p

    return marginal


def kl_from_uniform_latent(x, theta):
    n_enc = num_encoder_params()
    theta_enc = theta[:n_enc]

    qc = QuantumCircuit(TOTAL_QUBITS)
    enc, _ = encoder_circuit(x, theta_enc, NUM_QUBITS, LATENT_QUBITS)
    qc.compose(enc, inplace=True)

    sv = Statevector.from_instruction(qc)
    full_probs = sv.probabilities()

    n_data_states = 1 << NUM_QUBITS
    n_latent_states = 1 << LATENT_QUBITS
    latent_probs = np.zeros(n_latent_states)
    for i, p in enumerate(full_probs):
        latent_bits = i >> NUM_QUBITS
        latent_probs[latent_bits] += p

    uniform = 1.0 / n_latent_states
    kl = 0.0
    for p in latent_probs:
        if p > 0:
            kl += p * np.log(p / uniform)
    return kl


def train_qvae(target):
    n_states = 1 << NUM_QUBITS
    n_total_params = num_encoder_params() + num_decoder_params()
    theta0 = np.random.uniform(0, 2 * np.pi, n_total_params)

    x_samples = []
    for v in range(n_states):
        if target[v] > 1e-8:
            x = np.zeros(NUM_QUBITS)
            for i in range(NUM_QUBITS):
                x[i] = ((v >> i) & 1) * np.pi
            x_samples.append((x, target[v]))

    def cost(theta):
        recon_loss = 0.0
        kl_loss = 0.0
        for x, w in x_samples:
            output = forward(x, theta)
            for i, pt in enumerate(target):
                if pt > 0:
                    po = max(output[i], 1e-10)
                    recon_loss += w * pt * np.log(pt / po)
            kl_loss += w * kl_from_uniform_latent(x, theta)
        return float(recon_loss + BETA_KL * kl_loss)

    res = scipy_minimize(cost, theta0, method='COBYLA',
                         options={'maxiter': MAXITER, 'rhobeg': 0.5})
    return res.x, res.fun


def generate(theta):
    qc = QuantumCircuit(TOTAL_QUBITS)

    for i in range(NUM_QUBITS, TOTAL_QUBITS):
        qc.h(i)

    n_enc = num_encoder_params()
    theta_dec = theta[n_enc:]
    dec, _ = decoder_circuit(theta_dec, NUM_QUBITS, LATENT_QUBITS)
    qc.compose(dec, inplace=True)

    sv = Statevector.from_instruction(qc)
    full_probs = sv.probabilities()

    n_data_states = 1 << NUM_QUBITS
    marginal = np.zeros(n_data_states)
    for i, p in enumerate(full_probs):
        data_bits = i % n_data_states
        marginal[data_bits] += p

    return marginal


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- QVAE ({NUM_QUBITS}+{LATENT_QUBITS}q = {TOTAL_QUBITS}q, "
          f"{NUM_LAYERS} dec sloja, COBYLA {MAXITER} iter) ---")
    print(f"  Encoder params: {num_encoder_params()}, "
          f"Decoder params: {num_decoder_params()}")

    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        target = build_empirical(draws, pos)

        theta, loss = train_qvae(target)
        gen_dist = generate(theta)
        gen_dist = gen_dist - gen_dist.min()
        if gen_dist.sum() > 0:
            gen_dist /= gen_dist.sum()
        dists.append(gen_dist)

        top_idx = np.argsort(gen_dist)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{gen_dist[i]:.3f}" for i in top_idx)
        print(f"loss={loss:.4f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QVAE, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QVAE (5+2q = 7q, 2 dec sloja, COBYLA 250 iter) ---
  Encoder params: 14, Decoder params: 28
  Poz 1... loss=0.6801  top: 27:0.082 | 8:0.078 | 25:0.076
  Poz 2... loss=0.4790  top: 3:0.078 | 15:0.071 | 14:0.063
  Poz 3... loss=0.1993  top: 14:0.112 | 13:0.102 | 23:0.098
  Poz 4... loss=0.0725  top: 10:0.081 | 28:0.071 | 6:0.068
  Poz 5... loss=0.2653  top: 11:0.084 | 29:0.084 | 34:0.078
  Poz 6... loss=0.4950  top: 29:0.076 | 19:0.070 | 14:0.070
  Poz 7... loss=0.5622  top: 7:0.124 | 23:0.096 | 38:0.089

==================================================
Predikcija (QVAE, deterministicki, seed=39):
[27, 33, x, y, z, 37, 38]
==================================================
"""



"""
QVAE - Quantum Variational Autoencoder

7 qubita ukupno: 5 data qubita + 2 latentna qubita
Encoder: kompresuje ulaz u latentni prostor (2 qubita)
Decoder: rekonstruise distribuciju iz latentnog prostora (2 sloja Ry+CX+Rz)
VAE loss: rekonstrukcioni KL + beta * latentni KL (regularizacija ka uniformnom latentnom prostoru)
Generacija: H na latentnim qubitima → decoder → marginalna distribucija preko data qubita
Uci kompaktnu reprezentaciju podataka i generise iz nje
COBYLA 250 iteracija, deterministicki
Bez iterativnog treniranja kola - samo jednom se generisu parametri
"""

