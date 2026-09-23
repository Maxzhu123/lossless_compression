"""Reusable Matplotlib styling and axes-based plotting components.

Nothing here reads data; callers pass series in and keep control of labels,
legends, layout and saving. :func:`plot_style` is the one entry point that
changes global state, and it restores it on exit, so importing this module has
no styling side effects.

Copied from bitsparse/plots/plot_lib.py; series names adapted for LCT.
"""

import colorsys
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


# The base rcParams every figure starts from.
PLOT_PARAMS = {
    "figure.figsize": (5.5, 3.4),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#262626",
    "axes.labelcolor": "#262626",
    "axes.labelsize": 13,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.xmargin": 0.025,
    "axes.ymargin": 0.06,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    # The default tick padding is half the tick-mark length, which leaves the
    # numbers touching the marks. This gap clears them.
    "xtick.major.pad": 7.0,
    "ytick.major.pad": 7.0,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# --------------------------------------------------------------------------- #
# Series appearance
# --------------------------------------------------------------------------- #

NEUTRAL_COLOR = "#333333"

# Appearance per configuration, keyed by the name that appears in the legend.
# Figures look their series up here rather than taking a style by column
# position, so a configuration keeps one colour and marker wherever it is
# plotted. The hues are Okabe-Ito, which stay distinct for colourblind readers,
# and the markers stay distinguishable in greyscale.
CONFIG_STYLES = {
    "Dense": {"color": NEUTRAL_COLOR, "marker": "o"},
    "LCT": {"color": "#0072B2", "marker": "s"},
    "LCT-buffer": {"color": "#D55E00", "marker": "^"},
    "DFloat11": {"color": "#009E73", "marker": "D"},
    "SplitZip": {"color": "#CC79A7", "marker": "P"},
    "ZipNN": {"color": "#56B4E9", "marker": "v"},
    "nvCOMP-Bitcomp": {"color": "#E69F00", "marker": "o"},
    "nvCOMP-ANS": {"color": "#333333", "marker": "^"},
}

# Line styles cycled across metrics when one figure carries several per group.
LINE_STYLES = ("-", "--", "-.", ":")


# --------------------------------------------------------------------------- #
# Figure geometry
# --------------------------------------------------------------------------- #

WIDE_FIGURE_SIZE = (8, 4.5)
# Margin between the figure edge and the axes, as a fraction of the font size.
LAYOUT_PAD = 0.5

# Point sizes on the rcParams that carry text, which plot_style scales together.
FONT_KEYS = (
    "font.size",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "legend.title_fontsize",
)
# A wide figure keeps the same point sizes as a narrow one, so its text looks
# comparatively small once the figure is scaled to fit a page column. Applying
# this scale restores the balance.
WIDE_FONT_SCALE = 1.5


# --------------------------------------------------------------------------- #
# Legend spacing
# --------------------------------------------------------------------------- #

# Legends are kept tight so they read as attached to their plot: a short handle
# keeps each symbol close to its label instead of centring it in a wide empty
# box, and trimming the padding stops blank space from inflating the legend. The
# handle length also sets how far a companion row, such as a fitted equation, is
# indented under the row above it.
LEGEND_HANDLE_LENGTH = 1.0
LEGEND_HANDLE_TEXT_PAD = 0.3
LEGEND_BORDER_PAD = 0.3
# How far a legend placed beside the axes sits from them, as an axes fraction.
LEGEND_SIDE_ANCHOR = 1.005


# --------------------------------------------------------------------------- #
# Ordered-group ramp
# --------------------------------------------------------------------------- #

# Ordered groups such as network layers walk the hue wheel from red to violet, so
# the colour advances with the group's index. The walk deliberately stops short
# of the full circle: wrapping 360 degrees puts the last group only 1/count back
# from the first, making the two neighbours on the wheel so they read as the same
# red. Ending at three quarters of the wheel instead puts them at opposite ends
# of the spectrum while still covering the rainbow. Lightness cycles through the
# contrast levels independently of hue, so neighbouring groups differ in tone as
# well as in colour. Each tone's lightness is solved to hold its target contrast
# against white, and 3.0 is the WCAG minimum for graphical objects.
GROUP_SATURATION = 0.85
GROUP_CONTRAST_LEVELS = (3.0, 4.6, 6.6)
GROUP_HUE_SPAN = 0.75


def _relative_luminance(rgb):
    """Return the WCAG relative luminance of an ``(r, g, b)`` triple in 0-1."""
    def linearize(channel):
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _color_at_hue(hue, saturation, target):
    """Return the ``(r, g, b)`` at ``hue`` whose luminance is ``target``.

    Lightness is found by bisection. A hue that cannot reach the target even at
    the lightness ceiling, as saturated blues and reds cannot, is returned at
    that ceiling so it stays as rich as possible instead of turning pale.
    """
    low, high = 0.0, 0.5
    for _ in range(30):
        middle = (low + high) / 2
        if _relative_luminance(colorsys.hls_to_rgb(hue, middle, saturation)) > target:
            high = middle
        else:
            low = middle
    return colorsys.hls_to_rgb(hue, (low + high) / 2, saturation)


def plot_style(overrides=None, *, wide=False, font_scale=1.0):
    """Return a style context; use around figure creation, plotting and saving.

    ``overrides`` accepts Matplotlib rcParams. ``font_scale`` multiplies every
    text size, which is how a wide figure keeps its text legible at page scale.
    Settings are restored when the context exits.
    """
    params = {**PLOT_PARAMS}
    if wide:
        params["figure.figsize"] = WIDE_FIGURE_SIZE
    if font_scale != 1.0:
        params.update({key: params[key] * font_scale for key in FONT_KEYS})
    return plt.rc_context({**params, **(overrides or {})})


def plot_series(ax, series, *, styles=None, **line_kwargs):
    """Plot an iterable of ``(label, x_values, y_values)`` on an existing axes.

    ``styles`` maps a label to its style, defaulting to :data:`CONFIG_STYLES`, so
    a named series looks the same in every figure. Line keyword arguments
    override the shared styles. Return the created lines; the caller controls
    labels, legends, layout and saving.
    """
    styles = CONFIG_STYLES if styles is None else styles

    lines = []
    for label, x_values, y_values in series:
        if label not in styles:
            raise ValueError(f"no style defined for series {label!r}")
        options = {"markerfacecolor": "white", **styles[label], **line_kwargs}
        lines.extend(ax.plot(x_values, y_values, label=label, **options))
    return lines


def sample_group_colors(count, *, saturation=GROUP_SATURATION,
                        contrast_levels=GROUP_CONTRAST_LEVELS,
                        hue_span=GROUP_HUE_SPAN):
    """Return ``count`` vivid colours ordered by the group's position.

    Hue walks from red to violet, so the first and last group land at opposite
    ends of the spectrum rather than next to each other on the wheel, and
    neighbouring groups differ in tone as well as in hue. Every colour is legible
    on white. See the ramp constants above for the reasoning.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    levels = tuple(contrast_levels)
    if not levels:
        raise ValueError("contrast_levels must contain at least one value")

    offsets = [0.5] if count == 1 else [index / (count - 1) for index in range(count)]
    return [
        _color_at_hue(
            hue_span * offset, saturation, 1.05 / levels[index % len(levels)] - 0.05,
        )
        for index, offset in enumerate(offsets)
    ]


def plot_grouped_series(ax, x, groups, metrics, *, colors=None):
    """Plot ``{group: {metric: values}}`` with one colour per group.

    ``metrics`` is an ordered iterable of metric names, which take the cycled
    line styles in order and so must be supplied by every group. Groups take
    ``colors`` in order, defaulting to the ramp from :func:`sample_group_colors`
    so the colour reflects the group's position. Return the group and metric
    legend handles for :func:`finish_plot`; data parsing stays with the caller.
    """
    metric_styles = list(zip(metrics, cycle(LINE_STYLES)))
    items = list(groups.items())
    colors = sample_group_colors(len(items)) if colors is None else list(colors)
    if len(colors) != len(items):
        raise ValueError(f"expected {len(items)} colors, got {len(colors)}")

    group_handles = []
    for (label, values), color in zip(items, colors):
        for metric, linestyle in metric_styles:
            title = f"{label} {metric}"
            plot_series(
                ax, [(title, x, values[metric])],
                styles={title: {"color": color, "linestyle": linestyle}},
            )
        group_handles.append(Line2D([], [], color=color, label=label))
    metric_handles = [
        Line2D([], [], color=NEUTRAL_COLOR, linestyle=style, label=metric)
        for metric, style in metric_styles
    ]
    return group_handles, metric_handles


def format_axes(ax, *, xlabel, ylabel, xformat="{x:,.0f}", yformat=None, x_step=None):
    """Apply axis labels and optional Matplotlib number-format strings.

    ``x_step`` pins the x ticks to a fixed interval. Setting it is how a wide
    figure keeps its x labels apart: the automatic locator thins ticks from the
    rcParams font size, which rises with ``font_scale``, so its choice cannot be
    relied on to leave room.
    """
    ax.set(xlabel=xlabel, ylabel=ylabel)
    for axis, pattern in ((ax.xaxis, xformat), (ax.yaxis, yformat)):
        if pattern is not None:
            axis.set_major_formatter(StrMethodFormatter(pattern))
    if x_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_step))


def finish_plot(ax, *, group_handles=None, metric_handles=None, group_title=None,
                legend_outside=False, legend_entries=None):
    """Apply shared legend placement and layout to an axes' figure.

    Ordinary series take an inside legend, or one beside the axes when
    ``legend_outside`` is set, which needs ``plot_style(wide=True)`` to have room.
    ``legend_entries`` supplies an explicit ``(handles, labels)`` pair for a
    legend whose rows are not all plotted artists, such as one annotating each
    series with an extra line of text. Grouped series take a vertical legend and
    an optional metric key in reserved space on the right.
    """
    spacing = {
        "borderpad": LEGEND_BORDER_PAD,
        "handlelength": LEGEND_HANDLE_LENGTH,
        "handletextpad": LEGEND_HANDLE_TEXT_PAD,
    }

    if group_handles is None:
        if legend_outside:
            options = {
                "loc": "center left",
                "bbox_to_anchor": (LEGEND_SIDE_ANCHOR, 0.5),
                "borderaxespad": 0,
                **spacing,
            }
        else:
            options = {"loc": "best", "borderaxespad": 1.0, **spacing}

        if legend_entries is None:
            ax.legend(**options)
        else:
            ax.legend(*legend_entries, **options)
        ax.figure.tight_layout(pad=LAYOUT_PAD)
        return

    legend = ax.legend(
        handles=group_handles, title=group_title, loc="upper left",
        bbox_to_anchor=(LEGEND_SIDE_ANCHOR, 1), ncol=1, borderaxespad=0, **spacing,
    )
    if metric_handles is not None:
        ax.add_artist(legend)
        ax.legend(
            handles=metric_handles, loc="lower left",
            bbox_to_anchor=(LEGEND_SIDE_ANCHOR, 0), borderaxespad=0, **spacing,
        )
    ax.figure.tight_layout(pad=LAYOUT_PAD)
