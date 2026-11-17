# Benchmark visualizations

The data in `benchmark_scores.json` is a snapshot of the ALIVE paper's reported comparisons. It is not a live leaderboard. FSI-Bench uses the Explicit Decision Protocol (EDP), and all bars start at zero.

To regenerate the figures from the repository root:

```bash
python -m pip install matplotlib pillow numpy
python docs/visuals/render_benchmarks.py
```

The script writes PNG, SVG, and GIF versions to `docs/assets/`. GIFs animate once and then remain on the final scores. The README links each animation to its full-resolution static PNG. All four compared models use consistent colors in both benchmark groups.

The Paper button links to https://arxiv.org/pdf/2611.21366. Model and Dataset buttons in the main README intentionally have no links yet. To activate them, wrap each corresponding image in an anchor with its release URL and replace the SOON artwork.
