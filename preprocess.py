import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    eeg = commands.add_parser("eeg")
    eeg.add_argument("--input", type=Path, required=True)
    eeg.add_argument("--labels", type=Path, required=True)
    eeg.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from compensatory_audio_eeg.preprocess_eeg import preprocess
    preprocess(args.input, args.labels, args.output)


if __name__ == "__main__":
    main()
