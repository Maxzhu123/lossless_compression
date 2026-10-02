"""Launch configurations for TileLang decoder autotuning."""

# Retain the three measured winners and two lower-thread-count alternatives.
# Larger tiles reduce barriers; 128 threads give each thread two streams.
DECODE_AUTOTUNE_CONFIGS = (
    {"threads": 256, "row_tile": 8, "unroll": 4},
    {"threads": 256, "row_tile": 16, "unroll": 8},
    {"threads": 256, "row_tile": 32, "unroll": 8},
    {"threads": 128, "row_tile": 8, "unroll": 4},
    {"threads": 128, "row_tile": 32, "unroll": 8},
)
