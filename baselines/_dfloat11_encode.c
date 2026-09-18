/* Local DFloat11-compatible CPU encoder. The upstream decoder is unchanged. */
#include <stdint.h>
#include <stddef.h>

void histogram(const uint16_t *input, size_t n, uint64_t *counts) {
    for (size_t i = 0; i < n; ++i) ++counts[(input[i] >> 7) & 255];
}

/* Five-bit gaps are packed most-significant bit first, as in np.packbits. */
static void put_gap(uint8_t *gaps, size_t index, unsigned value) {
    size_t bit = index * 5;
    unsigned shift = bit % 8;
    uint16_t pair = (uint16_t)value << (11 - shift);
    gaps[bit / 8] |= pair >> 8;
    gaps[bit / 8 + 1] |= pair & 255;
}

size_t encode(const uint16_t *input, size_t n, const uint8_t *lengths,
              const uint32_t *codes, unsigned eof_length, uint32_t eof_code,
              uint8_t *encoded, uint8_t *sm, uint32_t *positions,
              uint8_t *gaps) {
    uint64_t buffer = 0, total_bits = 0;
    unsigned pending = 0;
    size_t out = 0, gap_count = 0, position_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (total_bits / 64 >= gap_count)
            put_gap(gaps, gap_count++, total_bits % 64);
        if (total_bits / 32768 >= position_count)
            positions[position_count++] = (uint32_t)i;
        uint16_t raw = input[i];
        unsigned symbol = (raw >> 7) & 255;
        unsigned length = lengths[symbol];
        sm[i] = ((raw >> 8) & 128) | (raw & 127);
        buffer = (buffer << length) | codes[symbol];
        pending += length;
        total_bits += length;
        while (pending >= 8) {
            pending -= 8;
            encoded[out++] = (uint8_t)(buffer >> pending);
        }
        buffer &= (UINT64_C(1) << pending) - 1;
    }
    if (pending) {
        if (total_bits / 64 >= gap_count)
            put_gap(gaps, gap_count++, total_bits % 64);
        if (total_bits / 32768 >= position_count)
            positions[position_count++] = (uint32_t)n;
        buffer = (buffer << eof_length) | eof_code;
        pending += eof_length;
        encoded[out++] = pending >= 8 ? buffer >> (pending - 8) : buffer << (8 - pending);
    }
    positions[position_count] = (uint32_t)n;
    return out;
}
