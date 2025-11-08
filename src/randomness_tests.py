import numpy as np
from scipy.special import erfc

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
