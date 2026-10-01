"""Decoder search space; every configuration preserves the stored format."""

# Keep the validated default first. Larger tiles trade fewer barriers for more
# shared memory; fewer threads give each thread multiple independent streams.
DEFAULT_DECODE_CONFIG = {"threads": 256, "row_tile": 8, "unroll": 4}
DECODE_AUTOTUNE_CONFIGS = (
    DEFAULT_DECODE_CONFIG,
    *(
        {"threads": threads, "row_tile": rows, "unroll": unroll}
        for threads in (64, 128, 256)
        for rows, unroll in ((4, 2), (8, 1), (8, 4), (16, 4), (16, 8), (32, 8))
        if (threads, rows, unroll) != (256, 8, 4)
    ),
)
