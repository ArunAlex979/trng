from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF
from Crypto.Random import get_random_bytes

class Conditioner:
    """
    Conditions the raw entropy to produce cryptographically secure random keys.
    """

    def __init__(self, key_size=32, salt=None):
        """
        Initializes the Conditioner.

        Args:
            key_size (int): The desired output key size in bytes (e.g., 32 for a 256-bit key).
            salt (bytes): An optional salt for the HKDF. If None, a random one is generated.
        """
        self.key_size = key_size
        if salt is None:
            self.salt = get_random_bytes(16)
        else:
            self.salt = salt

    def condition_data(self, entropy_pool):
        """
        Conditions the entropy pool to generate a secure key.

        Args:
            entropy_pool (bytearray): The raw entropy data.

        Returns:
            bytes: A cryptographically secure key.
        """
        # 1. Whiten the data using SHA-256
        hashed_entropy = SHA256.new(entropy_pool).digest()

        # 2. Use HKDF to derive the final key
        # The salt can be updated with each run for better security
        self.salt = SHA256.new(self.salt).digest() 

        key = HKDF(
            master=hashed_entropy,
            key_len=self.key_size,
            salt=self.salt,
            hashmod=SHA256,
            num_keys=1,
            context=b'fish-trng-key-v1'
        )
        
        return key

if __name__ == '__main__':
    # Example usage
    conditioner = Conditioner()

    # Simulate an entropy pool (should be larger in a real scenario)
    # For this example, we'll use 64 bytes of "random" data
    raw_entropy = b'\x1a\x2b\x3c\x4d\x5e\x6f\x7a\x8b' * 8
    
    print(f"Raw entropy (first 16 bytes): {raw_entropy[:16].hex()}...")
    print(f"Raw entropy length: {len(raw_entropy)} bytes")

    # Generate a key
    secure_key = conditioner.condition_data(raw_entropy)

    print(f"\nGenerated Secure Key (256-bit): {secure_key.hex()}")
    print(f"Key length: {len(secure_key)} bytes")

    # Generate another key with the same entropy to show the salt is changing
    # In a real application, the entropy pool would be different each time
    print("\n--- Generating another key with the same entropy ---")
    secure_key_2 = conditioner.condition_data(raw_entropy)
    print(f"Generated Secure Key 2 (256-bit): {secure_key_2.hex()}")

    # Note: The keys will be different because the salt is updated internally
    assert secure_key != secure_key_2
    print("\nSuccessfully generated two different keys from the same input due to salting.")
