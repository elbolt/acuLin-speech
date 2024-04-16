import argparse


def parse_arguments(default_subjects: list) -> list:
    """ Argument parser for parsing from command-line.  """
    parser = argparse.ArgumentParser(description='Process EEG for later analysis')
    parser.add_argument('-s', '--subjects', type=str, help='Subject numbers separated by commas (no spaces)')
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects.split(',')
        subjects = [subject.strip() for subject in subjects]
    else:
        subjects = default_subjects

    return subjects
