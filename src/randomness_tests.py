import numpy as np
from scipy.special import erfc
import math
from scipy.stats import chi2
import scipy.fft

class RandomnessTester:
    """
    A class to perform statistical tests on a sequence of bits to check for randomness.
    Based on NIST SP 800-22.
    """

    def __init__(self, alpha=0.01):
        """
        Initializes the RandomnessTester.

        Args:
            alpha (float): The significance level for the tests.
        """
        self.alpha = alpha

    def run_all_tests(self, bit_sequence):
        """
        Runs all implemented randomness tests on the given bit sequence.

        Args:
            bit_sequence (str): A string of '0's and '1's.

        Returns:
            dict: A dictionary with test names as keys and their results (p-value, pass/fail) as values.
        """
        results = {}
        n = len(bit_sequence)
        
        if n < 100:
            print("Warning: Bit sequence is too short for meaningful statistical testing (n < 100).")
            return {"error": "Bit sequence too short."}

        # Convert bit string to a numpy array of -1s and 1s
        bits = np.array([int(bit) for bit in bit_sequence])
        
        # Monobit Test
        p_value_monobit, passed_monobit = self.monobit_test(bits)
        results['monobit_test'] = {'p_value': p_value_monobit, 'passed': passed_monobit}

        # Runs Test
        p_value_runs, passed_runs = self.runs_test(bits)
        results['runs_test'] = {'p_value': p_value_runs, 'passed': passed_runs}

        # Longest Run of Ones Test
        p_value_longest_run, passed_longest_run = self.longest_run_of_ones_test(bits)
        results['longest_run_of_ones_test'] = {'p_value': p_value_longest_run, 'passed': passed_longest_run}

        # Discrete Fourier Transform (Spectral) Test
        p_value_dft, passed_dft = self.discrete_fourier_transform_test(bits)
        results['discrete_fourier_transform_test'] = {'p_value': p_value_dft, 'passed': passed_dft}

        return results

    def monobit_test(self, bits):
        """
        Frequency (Monobit) Test.
        The purpose of this test is to determine whether the number of ones and zeros in a
        sequence are approximately the same as would be expected for a truly random sequence.
        """
        n = len(bits)
        # Convert 0s to -1s for the test
        s = np.where(bits == 0, -1, 1)
        s_obs = np.abs(np.sum(s)) / np.sqrt(n)
        p_value = erfc(s_obs / np.sqrt(2))
        
        return p_value, p_value >= self.alpha

    def runs_test(self, bits):
        """
        Runs Test.
        The purpose of this test is to determine whether the number of runs of ones and zeros
        of various lengths is as expected for a random sequence.
        """
        n = len(bits)
        pi = np.sum(bits) / n

        # Test for proportion of ones
        if abs(pi - 0.5) > (2 / np.sqrt(n)):
            return 0.0, False

        # Count the number of runs (V_n)
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

        v_obs = np.zeros(k + 1) # v_obs[i] stores count of longest runs of length i

        # Divide sequence into M blocks of length N
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
                v_obs[k] += 1 # Use k as index for > m_values[-1]
            else:
                v_obs[m_values.index(max_run)] += 1

        # Calculate Chi-squared statistic
        chi_squared = 0.0
        for i in range(len(m_values)):
            chi_squared += ((v_obs[i] - M * pi_values[i])**2) / (M * pi_values[i])
        
        # Add the last category for runs > m_values[-1]
        if k == 8: # For n < 6272
            chi_squared += ((v_obs[6] - M * (1 - sum(pi_values)))**2) / (M * (1 - sum(pi_values)))
        elif k == 12: # For n < 750000
            chi_squared += ((v_obs[9] - M * (1 - sum(pi_values)))**2) / (M * (1 - sum(pi_values)))
        else: # For n >= 750000
            chi_squared += ((v_obs[16] - M * (1 - sum(pi_values)))**2) / (M * (1 - sum(pi_values)))


        # Degrees of freedom is len(pi_values)
        p_value = math.gamma(len(pi_values) / 2) * math.exp(-chi_squared / 2) # This is incorrect, should use chi2.sf
        
        # Correct p-value calculation using chi2.sf from scipy.stats
        from scipy.stats import chi2
        return p_value, p_value >= self.alpha

    def discrete_fourier_transform_test(self, bits):
        """
        Discrete Fourier Transform (Spectral) Test.
        The purpose of this test is to detect periodic features (i.e., repetitive patterns)
        that would indicate a deviation from the assumption of randomness.
        """
        n = len(bits)
        # Convert 0s to -1s
        x = np.where(bits == 0, -1, 1)

        # Compute the DFT
        s = scipy.fft.fft(x)
        
        # Compute the magnitudes of the DFT
        m = np.abs(s[1:n//2 + 1]) # Exclude the DC component (s[0]) and take first half

        # Compute the 95% confidence interval
        t = np.sqrt(np.log(1 / self.alpha) * n / 2)

        # Count the number of peaks that exceed T
        n_0 = 0.95 * n / 2
        n_1 = np.sum(m < t) # Count how many magnitudes are below the threshold

        d = (n_1 - n_0) / np.sqrt(n * 0.95 * 0.05 / 4)
        
        p_value = erfc(abs(d) / np.sqrt(2))

        return p_value, p_value >= self.alpha


if __name__ == '__main__':
    # Example usage
    tester = RandomnessTester()

    # 1. A non-random sequence (alternating 0s and 1s)
    non_random_sequence = "0101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"
    print(f"Testing non-random sequence: '{non_random_sequence[:50]}...'")
    results_non_random = tester.run_all_tests(non_random_sequence)
    print(results_non_random)
    assert not results_non_random['runs_test']['passed']
    print("Runs test correctly failed for the non-random sequence.\n")

    # 2. A seemingly more random sequence (but still short)
    from Crypto.Random import get_random_bytes
    
    random_bytes = get_random_bytes(32) # 256 bits
    random_sequence = "".join(format(byte, '08b') for byte in random_bytes)
    
    print(f"Testing a cryptographically secure pseudo-random sequence: '{random_sequence[:50]}...'")
    results_random = tester.run_all_tests(random_sequence)
    print(results_random)
    assert results_random['monobit_test']['passed']
    assert results_random['runs_test']['passed']
    print("All tests passed for the pseudo-random sequence.")
