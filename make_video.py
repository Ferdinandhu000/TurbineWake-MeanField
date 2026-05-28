from pathlib import Path

import imageio.v2 as imageio


def main() -> None:
    root = Path(__file__).resolve().parent
    plots_dir = root / "plots"

    # Update the glob if you want a narrower selection.
    frames = sorted(plots_dir.glob("*.png"))
    if not frames:
        raise SystemExit(f"No .png frames found in {plots_dir}")

    output_path = root / "vedio.mp4"
    fps = 50

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))

    print(f"Wrote {output_path} with {len(frames)} frames at {fps} fps")


if __name__ == "__main__":
    main()
