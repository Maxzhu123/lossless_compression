"""Plot codec throughput versus uncompressed input size.

Run from the repository root: python plots/plot_codec_throughput.py
The supplied size column (originally named tensor_bytes) is interpreted as
MiB: 1024 denotes the 1 GiB benchmark. Times are means in milliseconds.
LZ4 and Cascaded are omitted as requested. No uncertainty was supplied.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from plot_lib import (
    LAYOUT_PAD,
    LEGEND_BORDER_PAD,
    LEGEND_HANDLE_LENGTH,
    LEGEND_HANDLE_TEXT_PAD,
    format_axes,
    plot_series,
)
from plot_tables import MIB_PER_GIB, parse_grouped, render


# Each method has adjacent encode/decode mean times, both in ms.
TIMINGS = """
nvCOMP-Bitcomp nvCOMP-ANS DFloat11 ZipNN SplitZip LCT
size_mib encode decode encode decode encode decode encode decode encode decode encode decode
1    1.119 1.043 1.184 1.058 2.780    0.078 0.524    0.336   0.146 0.066 0.302 0.146
2    1.102 1.076 1.155 0.983 5.083    0.121 0.789    0.456   0.160 0.068 0.316 0.142
4    1.167 1.038 1.363 1.014 9.594    0.115 1.271    0.643   0.193 0.079 0.304 0.158
8    1.228 1.024 1.208 1.101 18.125   0.140 2.127    1.017   0.194 0.068 0.301 0.192
16   1.215 1.105 1.225 0.992 36.124   0.182 4.265    1.874   0.195 0.100 0.298 0.198
32   1.581 1.119 2.529 1.142 80.052   0.331 23.414   6.665   0.297 0.145 0.319 0.222
64   3.749 1.219 3.771 1.234 167.480  0.449 48.037   13.311  0.485 0.232 0.318 0.274
128  6.236 1.564 7.269 1.462 331.829  0.783 105.328  26.424  0.852 0.424 0.461 0.420
256  11.000 1.755 14.355 1.940 667.245 1.310 215.755 53.190 1.671 0.703 0.771 0.766
512  20.786 2.560 26.818 2.565 1330.822 2.667 430.583 103.383 3.200 1.370 1.340 1.393
1024 40.413 4.193 53.028 4.349 2626.546 5.217 845.402 199.273 6.416 2.704 2.613 2.725
2048 78.412 7.565 104.129 7.442 5283.261 9.808 1684.393 396.041 13.085 5.193 5.057 5.191
"""


def load_throughputs():
    sizes, timings = parse_grouped(
        TIMINGS, x_name="size_mib", metrics=("encode", "decode"),
    )
    # Use original input bytes for both operations, not compressed payload size.
    return {
        metric: [
            (name, sizes, [size / MIB_PER_GIB * 1000 / ms
                           for size, ms in zip(sizes, values[metric])])
            for name, values in timings.items()
        ]
        for metric in ("encode", "decode")
    }


def build(series):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax, metric in zip(axes, ("encode", "decode")):
        visible = []
        for name, sizes, speeds in series[metric]:
            points = [(size, speed) for size, speed in zip(sizes, speeds)
                      if size >= 16]
            x, y = zip(*points)
            visible.append((name, x, y))
        plot_series(ax, visible)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024, 2048])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, 425)
        ax.set_title(metric.capitalize())
        format_axes(
            ax, xlabel="Input size / MiB",
            ylabel="Throughput / GiB/s" if metric == "encode" else "",
            xformat="{x:.0f}", yformat="{x:g}",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=LAYOUT_PAD)
    fig.legend(
        handles, [label.replace("nvCOMP-", "nvCOMP ") for label in labels],
        loc="center left", ncol=1, bbox_to_anchor=(1.0, 0.5),
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        borderpad=LEGEND_BORDER_PAD,
    )
    return fig, axes


def main():
    render(
        {"codec_throughput.pdf": load_throughputs()}, build,
        output_dir=Path(__file__).resolve().parent / "plots",
        wide=True, show=False,
    )


if __name__ == "__main__":
    main()
