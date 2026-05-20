#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file> [output.png]")
        sys.exit(1)

    path = sys.argv[1]
    data = np.loadtxt(path, delimiter=",")
    print(f"Shape: {data.shape}, unique values: {np.unique(data)}")

    unique = np.unique(data)
    n = len(unique)
    color_map = ListedColormap(matplotlib.colormaps["hsv"](np.linspace(0, 0.9, max(n, 1))))

    fig, ax = plt.subplots(figsize=(10, 8))
    mapped = np.searchsorted(unique, data)
    im = ax.imshow(mapped, cmap=color_map, interpolation="nearest", aspect="auto")

    ax.set_title(path)
    plt.tight_layout()

    out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0] + ".png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
