"""Table parsing and figure rendering shared by the figure scripts.

Each figure script keeps its measurements in a text table so the numbers stay
visible in the diff. This module turns those tables into the series the plotting
helpers consume, and owns the render loop so no script reimplements saving.

Table headers double as series names, so they are read verbatim and must match
the keys in :data:`plot_lib.CONFIG_STYLES`; keeping them single tokens also lets
a table stay whitespace-separable with its gaps written out.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_lib import plot_style


# Written where a series was not measured at that x value. Gaps have to be
# spelled out: splitting on whitespace collapses an empty cell, which would
# silently shift the following values into the wrong column.
MISSING = "-"

# The benchmark harnesses report VRAM in MiB; the figures use GiB.
MIB_PER_GIB = 1024.0


def _rows(table):
    """Return the table as token lists, dropping blank lines."""
    rows = [line.split() for line in table.splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError("expected a header and at least one data row")
    return rows


def _require_unique(names, what):
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate {what}: {names}")


def _require_width(rows, width, first_line):
    """Require every row to have ``width`` tokens.

    A row of the wrong length means a gap was left blank rather than written as
    ``MISSING``, so this check is what stops a misaligned table from parsing
    silently into the wrong columns.
    """
    for offset, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"line {first_line + offset}: expected {width} values, got {len(row)}"
            )


def _value(token):
    """Parse one cell, allowing thousands separators."""
    return float(token.replace(",", ""))


def parse_series(table, *, x_name, y_scale=1.0):
    """Parse a table holding one column per series.

    The first line names the x column and then each series; every later line is
    one x value followed by a measurement for each series, using ``MISSING`` for
    gaps. Measurements are scaled by ``y_scale`` so datasets recorded in
    different units can share an axis.

    Return ``(label, x_values, y_values)`` per series, in column order.
    """
    rows = _rows(table)
    headers = rows[0]
    if headers[0] != x_name:
        raise ValueError(f"first column should be {x_name!r}, not {headers[0]!r}")
    _require_unique(headers, "column names")
    _require_width(rows[1:], len(headers), first_line=2)

    x_values = {name: [] for name in headers[1:]}
    y_values = {name: [] for name in headers[1:]}
    for row in rows[1:]:
        x = _value(row[0])
        for name, token in zip(headers[1:], row[1:]):
            if token == MISSING:
                continue
            x_values[name].append(x)
            y_values[name].append(_value(token) * y_scale)

    return [(name, x_values[name], y_values[name]) for name in headers[1:]]


def parse_grouped(table, *, x_name, metrics):
    """Parse a table holding several metrics per series.

    The first line names each series and the second names the columns: the x
    column followed by the metrics of series 0, then those of series 1, and so
    on. Grouping the columns keeps the measurements of one series adjacent,
    which is easier to check by eye than a flat table when they are a tradeoff.

    Return ``(x_values, {series: {metric: values}})``. The x values are shared
    by every column, so a caller whose first column is a row label rather than
    an axis simply ignores them.
    """
    rows = _rows(table)
    labels = rows[0]
    columns = rows[1]
    metric_names = list(metrics)

    if columns[0] != x_name:
        raise ValueError(f"first column should be {x_name!r}, not {columns[0]!r}")
    _require_unique(labels, "series names")
    _require_width(rows[1:], 1 + len(labels) * len(metric_names), first_line=2)

    x_values = [_value(row[0]) for row in rows[2:]]
    series = {}
    for index, label in enumerate(labels):
        entry = {}
        for offset, metric in enumerate(metric_names):
            column = 1 + index * len(metric_names) + offset
            entry[metric] = [
                _value(row[column]) for row in rows[2:] if row[column] != MISSING
            ]
        series[label] = entry
    return x_values, series


def render(datasets, build, *, output_dir, wide=False, font_scale=1.0, show=True):
    """Save one PDF per dataset, optionally showing the figures.

    ``datasets`` maps an output filename to whatever ``build`` accepts, and
    ``build`` returns ``(figure, axes)``. The style context wraps building,
    saving and showing alike, so the figure is created with the intended sizes
    and ``plt.show`` does not redraw it under different ones. ``wide`` and
    ``font_scale`` are forwarded to :func:`~plot_lib.plot_style`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plot_style(wide=wide, font_scale=font_scale):
        for filename, dataset in datasets.items():
            fig, _ = build(dataset)
            fig.savefig(Path(output_dir) / filename, format="pdf")
            if not show:
                plt.close(fig)
        if show:
            plt.show()
