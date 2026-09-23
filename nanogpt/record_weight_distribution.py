"""Record nanoGPT weight exponents with the same early-training setup as momentum."""
from record_momentum_distribution import main


if __name__ == "__main__":
    main(kind="weights")
