"""CLI: GRID .mpg + .align → grayscale mouth-ROI .npz clips.

Usage on Colab / Kaggle:
    python scripts/preprocess.py \
        --video-dir /content/drive/MyDrive/grid_videos \
        --align-dir /content/drive/MyDrive/grid_align \
        --output-dir /content/drive/MyDrive/LSN_Data/grid_processed_new \
        --landmark-path shape_predictor_68_face_landmarks.dat \
        --speakers s1 s2 s3 s4 s5

Download the landmark model first:
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
"""

import argparse
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess GRID videos into .npz mouth-ROI clips"
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Root dir with speaker subdirs (s1/, s2/, ...) containing .mpg files",
    )
    parser.add_argument(
        "--align-dir",
        required=True,
        type=Path,
        help="Root dir with speaker subdirs (s1/, s2/, ...) containing .align files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination root; speaker subdirs are created automatically",
    )
    parser.add_argument(
        "--landmark-path",
        required=True,
        type=Path,
        help="Path to shape_predictor_68_face_landmarks.dat",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=["s1", "s2", "s3", "s4", "s5"],
        help="Speaker IDs to process (default: s1 s2 s3 s4 s5)",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=75,
        help="Frames per clip after resampling (default: 75)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    if not args.landmark_path.exists():
        raise FileNotFoundError(
            f"Landmark model not found: {args.landmark_path}\n"
            "Download with:\n"
            "  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            "  bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2"
        )

    import dlib

    from lsn.preprocessing import process_speaker

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.landmark_path))

    total_ok = total_skip = 0
    for speaker in args.speakers:
        log.info("Processing %s ...", speaker)
        n_ok, n_skip = process_speaker(
            args.video_dir,
            args.align_dir,
            args.output_dir,
            speaker,
            detector,
            predictor,
            target_frames=args.target_frames,
        )
        log.info("  %s: %d saved, %d skipped", speaker, n_ok, n_skip)
        total_ok += n_ok
        total_skip += n_skip

    log.info("Done. Total: %d clips saved, %d skipped.", total_ok, total_skip)


if __name__ == "__main__":
    main()
