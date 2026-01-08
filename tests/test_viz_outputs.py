from pathlib import Path

import numpy as np

from cryptomvp.viz.plotting import (
    plot_bar,
    plot_confusion_matrix,
    plot_histogram,
    plot_series_with_band,
    plot_series_with_mean_band,
)


def test_viz_outputs(tmp_path: Path):
    x = np.arange(10)
    y = np.random.normal(0, 1, size=10)
    out_base = tmp_path / "series"
    plot_series_with_band(
        x,
        y,
        window=3,
        title="Test Series",
        xlabel="x",
        ylabel="y",
        label="y",
        out_base=out_base,
        formats=["png"],
    )
    plot_series_with_mean_band(
        x,
        y,
        window=3,
        title="Test Series Alias",
        xlabel="x",
        ylabel="y",
        label="y",
        out_base=tmp_path / "series_alias",
        formats=["png"],
    )

    plot_histogram(
        y,
        bins=5,
        title="Hist",
        xlabel="x",
        ylabel="count",
        out_base=tmp_path / "hist",
        formats=["png"],
    )

    plot_bar(
        ["a", "b"],
        [1, 2],
        title="Bar",
        xlabel="cat",
        ylabel="val",
        out_base=tmp_path / "bar",
        formats=["png"],
    )

    plot_confusion_matrix(
        np.array([[1, 2], [3, 4]]),
        labels=["0", "1"],
        title="CM",
        out_base=tmp_path / "cm",
        formats=["png"],
    )

    assert (tmp_path / "series.png").exists()
    assert (tmp_path / "series_alias.png").exists()
    assert (tmp_path / "hist.png").exists()
    assert (tmp_path / "bar.png").exists()
    assert (tmp_path / "cm.png").exists()
