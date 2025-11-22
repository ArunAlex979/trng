import numpy as np
from scipy.special import erfc
import math
import scipy.fft
from scipy.stats import chi2

class RandomnessTester:
    """
    Lightweight SP 800-22-inspired test suite: Monobit, Runs, Longest Run,
    Spectral (DFT), plus Block Frequency, Cumulative Sums, Approximate Entropy, Serial.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def run_all_tests(self, bit_sequence):
        results = {}
        n = len(bit_sequence)
        if n < 100:
            return {"error": "Bit sequence too short."}
        bits = np.array([int(bit) for bit in bit_sequence], dtype=np.uint8)

        pv, ok = self.monobit_test(bits); results['monobit_test'] = {'p_value': pv, 'passed': ok}
        pv, ok = self.runs_test(bits); results['runs_test'] = {'p_value': pv, 'passed': ok}
        pv, ok = self.longest_run_of_ones_test(bits); results['longest_run_of_ones_test'] = {'p_value': pv, 'passed': ok}
        pv, ok = self.discrete_fourier_transform_test(bits); results['discrete_fourier_transform_test'] = {'p_value': pv, 'passed': ok}

        pv, ok = self.block_frequency_test(bits, block_size=128); results['block_frequency_test'] = {'p_value': pv, 'passed': ok}
        pv, ok = self.cumulative_sums_test(bits); results['cumulative_sums_test'] = {'p_value': pv, 'passed': ok}
        pv, ok = self.approximate_entropy_test(bits, m=2); results['approximate_entropy_test'] = {'p_value': pv, 'passed': ok}
        pv, ok = self.serial_test(bits, m=2); results['serial_test'] = {'p_value': pv, 'passed': ok}

        return results

    def monobit_test(self, bits):
        n = len(bits)
        s = np.where(bits == 0, -1, 1)
        s_obs = np.abs(np.sum(s)) / np.sqrt(n)
        p_value = erfc(s_obs / np.sqrt(2))
        return p_value, p_value >= self.alpha

    def runs_test(self, bits):
        n = len(bits)
        pi = np.sum(bits) / n
        if abs(pi - 0.5) > (2 / np.sqrt(n)):
            return 0.0, False
        v_obs = np.sum(bits[:-1] != bits[1:]) + 1
        p_value = erfc(abs(v_obs - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))
        return p_value, p_value >= self.alpha

    def longest_run_of_ones_test(self, bits):
        """
        Longest Run of Ones Test.
        The purpose of this test is to determine whether the length of the longest run of ones
        within the sequence is consistent with that of a random sequence.
        """
        n = len(bits)
        if n < 128: # Minimum length for this test as per NIST SP 800-22
            return 0.0, False # Or handle as an error/warning

        # Determine block size based on n
        if n < 6272:
            k = 8
            m_values = [1, 2, 3, 4, 5, 6]
            pi_values = [0.1174, 0.2430, 0.2493, 0.1750, 0.1027, 0.0623]
        elif n < 750000:
            k = 12
            m_values = [4, 5, 6, 7, 8, 9]
            pi_values = [0.1170, 0.2460, 0.1740, 0.1170, 0.0930, 0.0690]
        else:
            k = 10000
            m_values = [10, 11, 12, 13, 14, 15, 16]
            pi_values = [0.0882, 0.1090, 0.1523, 0.1855, 0.1990, 0.1501, 0.1059]

        v_obs = np.zeros(len(pi_values)) # v_obs[i] stores count of longest runs of length i

        # Divide sequence into M blocks of length k
        M = n // k
        for i in range(M):
            block = bits[i*k : (i+1)*k]
            
            max_run = 0
            current_run = 0
            for bit in block:
                if bit == 1:
                    current_run += 1
                else:
                    max_run = max(max_run, current_run)
                    current_run = 0
            max_run = max(max_run, current_run) # Check for run at the end of the block

            # Map max_run to appropriate category
            if max_run < m_values[0]:
                v_obs[0] += 1
            elif max_run > m_values[-1]:
                v_obs[-1] += 1 
            else:
                v_obs[m_values.index(max_run)] += 1

        # Calculate Chi-squared statistic
        chi_squared = 0.0
        for i in range(len(pi_values)):
            chi_squared += ((v_obs[i] - M * pi_values[i])**2) / (M * pi_values[i])
        
        # Degrees of freedom is len(pi_values) - 1? No, it's K=len(pi_values) classes, so K-1?
        # NIST SP 800-22 says degrees of freedom = K = len(pi_values) - 1?
        # Actually, NIST says "degrees of freedom = K" where K is the number of classes?
        # Wait, NIST says "degrees of freedom = K" where K is the number of bins.
        # Let's double check. NIST SP 800-22 Rev 1a Section 2.4.4 says "degrees of freedom = K".
        # But typically chi-square goodness of fit is K-1.
        # The NIST document says "P-value = igamc(K/2, chi_obs/2)".
        # K is the number of degrees of freedom.
        # For K=8 (M=8), degrees of freedom is 3? No.
        # Let's stick to the standard implementation which usually uses K bins.
        # The standard implementation uses K=3 for M=8, K=5 for M=128, K=6 for M=10000.
        # Wait, the number of bins is len(pi_values).
        # So degrees of freedom = len(pi_values).
        
        # [FIX] Use chi2.sf (survival function) which is 1 - cdf
        # And degrees of freedom = len(pi_values)
        p_value = chi2.sf(chi_squared, len(pi_values))
        
        return p_value, p_value >= self.alpha

    def discrete_fourier_transform_test(self, bits):
        n = len(bits)
        x = np.where(bits == 0, -1, 1)
        s = scipy.fft.fft(x)
        m = np.abs(s[1:n//2 + 1])
        t = np.sqrt(np.log(1 / self.alpha) * n / 2)
        n_0 = 0.95 * n / 2
        n_1 = np.sum(m < t)
        d = (n_1 - n_0) / np.sqrt(n * 0.95 * 0.05 / 4)
        p_value = erfc(abs(d) / np.sqrt(2))
        return p_value, p_value >= self.alpha

    def block_frequency_test(self, bits, block_size=128):
        n = len(bits)
        if n < block_size:
            return 0.0, False
        num_blocks = n // block_size
        pvals = []
        for i in range(num_blocks):
            block = bits[i*block_size:(i+1)*block_size]
            ones = np.sum(block)
            pi = ones / block_size
            chi_sq = 4 * block_size * (pi - 0.5) ** 2
            p = math.exp(-chi_sq / 2)
            pvals.append(p)
        p_value = float(np.mean(pvals))
        return p_value, p_value >= self.alpha

    def cumulative_sums_test(self, bits):
        x = np.where(bits == 0, -1, 1)
        s = np.cumsum(x)
        z = np.max(np.abs(s))
        n = len(bits)
        p_value = erfc(z / (np.sqrt(2 * n)))
        return p_value, p_value >= self.alpha

    def approximate_entropy_test(self, bits, m=2):
        n = len(bits)
        def phi(m):
            counts = {}
            s = np.append(bits, bits[:m-1])
            for i in range(n):
                k = tuple(s[i:i+m])
                counts[k] = counts.get(k, 0) + 1
            probs = np.array(list(counts.values()), dtype=np.float64) / n
            return float(np.sum(probs * np.log(probs + 1e-12)))
        ap_en = phi(m) - phi(m + 1)
        p_value = math.exp(-abs(ap_en) * n)
        return p_value, p_value >= self.alpha

    def serial_test(self, bits, m=2):
        n = len(bits)
        s = np.append(bits, bits[:m-1])
        counts_m, counts_m1, counts_m2 = {}, {}, {}
        for i in range(n):
            k2 = tuple(s[i:i+2]); counts_m2[k2] = counts_m2.get(k2, 0) + 1
        for i in range(n):
            k = tuple(s[i:i+m]); counts_m[k] = counts_m.get(k, 0) + 1
        for i in range(n):
            k1 = tuple(s[i:i+m-1]); counts_m1[k1] = counts_m1.get(k1, 0) + 1

        def psi(counts):
            total = sum(counts.values())
            return sum((c/total) * np.log(c/total + 1e-12) for c in counts.values())

        psi_m = psi(counts_m)
        psi_m1 = psi(counts_m1)
        psi_m2 = psi(counts_m2)
        v = psi_m - psi_m1
        w = psi_m - psi_m2
        p1 = math.exp(-abs(v) * n)
        p2 = math.exp(-abs(w) * n)
        p_value = float(min(p1, p2))
        return p_value, p_value >= self.alpha

    def min_entropy_per_bit(self, bit_sequence):
        n = len(bit_sequence)
        if n == 0:
            return 0.0
        num_zeros = bit_sequence.count('0')
        num_ones = bit_sequence.count('1')
        max_prob = max(num_zeros / n, num_ones / n)
        return -math.log2(max_prob if max_prob > 0 else 1.0)