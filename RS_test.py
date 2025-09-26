import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import reedsolo as rs
import math

from sionna.phy.utils import ebnodb2no
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder

# =========================
# Parameters
# =========================
RS_N = 255
RS_K = 223
CODERATE_RS = RS_K / RS_N
BITS_PER_BYTE = 8

LDPC_N = 16384
LDPC_RATE = 0.5
LDPC_K = int(LDPC_N * LDPC_RATE)

NUM_BLOCKS = 200  # increase for smoother curves

# Eb/N0 range [dB]
ebn0_dbs = np.arange(0, 3, 1)  # -2 .. 6 dB

# =========================
# Components
# =========================
rs.init_tables(0x11D)

# constellation + mappers/demappers
constellation = Constellation("custom", 1, points=[-1, 1])
mapper = Mapper(constellation=constellation)

# Demappers: one hard for RS, one soft (LLR) for LDPC
demapper_hard = Demapper("app", constellation=constellation, hard_out=True)   # RS path (hard decisions)
demapper_soft = Demapper("app", constellation=constellation, hard_out=False)  # LDPC path (LLRs)

awgn = AWGN()

# LDPC (5G) encoder/decoder
ldpc_enc = LDPC5GEncoder(LDPC_K, LDPC_N)
ldpc_dec = LDPC5GDecoder(ldpc_enc, hard_out=True)

# =========================
# Channel abstraction -- central place to modify later
# =========================
def chanSim(tx, no, mode="awgn"):
    """
    CHANNEL ABSTRACTION (### MARKER ###)
    Currently: AWGN only.
    Replace or extend this function to add fading, ISI, MIMO, etc.
    """
    return awgn(tx, no)

# =========================
# Q-function (using erfc)
# =========================
def Q(x):
    return np.vectorize(lambda z: 0.5 * math.erfc(z / math.sqrt(2.0)))(x)

# =========================
# Simulation containers
# =========================
ber_uncoded = []
ber_rs = []
ber_ldpc = []

# =========================
# Main simulation loop
# =========================
for ebn0_db in ebn0_dbs:
    no_uncoded = ebnodb2no(ebn0_db, coderate=1, num_bits_per_symbol=1)
    no_rs = ebnodb2no(ebn0_db, coderate=CODERATE_RS, num_bits_per_symbol=1)
    no_ldpc = ebnodb2no(ebn0_db, coderate=LDPC_RATE, num_bits_per_symbol=1)

    bit_errs_uncoded = 0
    bit_errs_rs = 0
    bit_errs_ldpc = 0
    total_bits_uncoded = 0
    total_bits_rs = 0
    total_bits_ldpc = 0

    for _ in range(NUM_BLOCKS):
        # -------------------------
        # Uncoded BPSK path
        # -------------------------
        # generate random bits
        bits_uncoded = np.random.randint(0, 2, size=(LDPC_K,), dtype=np.uint8)  
        tf_bits_uncoded = tf.constant(bits_uncoded, dtype=tf.int32)
        
        # map bits to BPSK symbols
        tx_uncoded = mapper(tf.reshape(tf_bits_uncoded, (1, -1)))
        
        # channel (AWGN)
        rx_uncoded = chanSim(tx_uncoded, no=ebnodb2no(ebn0_db, coderate=1.0, num_bits_per_symbol=1))
        
        # demap (hard decisions)
        demapped_bits_uncoded = demapper_hard(rx_uncoded, ebnodb2no(ebn0_db, coderate=1.0, num_bits_per_symbol=1)).numpy().astype(np.uint8)[0]
        
        # count bit errors
        bit_errs_uncoded += np.sum(bits_uncoded != demapped_bits_uncoded)
        total_bits_uncoded += len(bits_uncoded)

        # -------------------------
        # RS(255,223) path
        # -------------------------
        msg = np.random.bytes(RS_K)
        rs_encoder = rs.RSCodec(RS_N - RS_K)
        codeword = rs_encoder.encode(msg)  # 255 bytes

        bits = np.unpackbits(np.frombuffer(codeword, dtype=np.uint8))
        tf_bits = tf.constant(bits, dtype=tf.int32)
        tx = mapper(tf.reshape(tf_bits, (1, -1)))

        rx = chanSim(tx, no_rs)

        demapped_bits_rs = demapper_hard(rx, no_rs).numpy().astype(np.uint8)[0]

        # BER (uncoded before RS decoding)
        bit_errs_uncoded += np.sum(bits != demapped_bits_rs)
        total_bits_uncoded += len(bits)

        # RS decode
        rx_bytes = np.packbits(demapped_bits_rs)
        try:
            decoded_msg, _, _ = rs_encoder.decode(rx_bytes)
            ref_bits = np.unpackbits(np.frombuffer(msg, dtype=np.uint8))
            dec_bits = np.unpackbits(np.frombuffer(decoded_msg, dtype=np.uint8))
            bit_errs_rs += np.sum(ref_bits != dec_bits)
        except rs.ReedSolomonError:
            bit_errs_rs += RS_K * BITS_PER_BYTE  # uncorrectable

        total_bits_rs += RS_K * BITS_PER_BYTE

        # -------------------------
        # LDPC path
        # -------------------------
        info_bits = tf.cast(tf.random.uniform([1, LDPC_K], maxval=2, dtype=tf.int32), tf.float32)
        code_bits = ldpc_enc(info_bits)

        tx_ldpc = mapper(tf.cast(code_bits, tf.int32))
        rx_ldpc = chanSim(tx_ldpc, no_ldpc)

        llrs = demapper_soft(rx_ldpc, no_ldpc)
        decoded_bits = ldpc_dec(llrs)

        ref = info_bits.numpy().astype(np.int32)[0]
        dec = decoded_bits.numpy()[0].astype(np.int32)
        bit_errs_ldpc += np.sum(ref != dec)
        total_bits_ldpc += LDPC_K

    ber_uncoded.append(bit_errs_uncoded / total_bits_uncoded)
    ber_rs.append(bit_errs_rs / total_bits_rs)
    ber_ldpc.append(bit_errs_ldpc / total_bits_ldpc)

    print(f"Eb/N0={ebn0_db} dB: uncoded={ber_uncoded[-1]:.3e}, RS={ber_rs[-1]:.3e}, LDPC={ber_ldpc[-1]:.3e}")

# =========================
# Theoretical uncoded BPSK
# =========================
#ebn0_lin = 10.0 ** (-ebn0_dbs / 20.0)
ber_theory = 0.5 * __import__("scipy.special").special.erfc(np.sqrt(10 ** (ebn0_dbs / 10)))

# =========================
# Plotting
# =========================
plt.figure(figsize=(8,5))
plt.semilogy(ebn0_dbs, ber_uncoded, 'o-', label="BPSK empirical (uncoded)")
plt.semilogy(ebn0_dbs, ber_theory, 'k--', label="BPSK theory (Q-function)")
plt.semilogy(ebn0_dbs, ber_rs, 's-', label="RS(255,223) empirical")
plt.semilogy(ebn0_dbs, ber_ldpc, 'd-', label=f"LDPC ({LDPC_N}, R={LDPC_RATE}) empirical")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs Eb/N0 (empirical + theory)")
plt.grid(True, which="both")
plt.legend()
plt.ylim(1e-6, 1)
plt.show()
