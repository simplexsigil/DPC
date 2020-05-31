import argparse
import csv
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--action_folder', default='', type=str, help='path of folder containing actions')
parser.add_argument('--csv_path', default='', type=str, help='path where csv file shall be stored.')


def main():
    """
    Uses directory names as action class labels and writes them to a csv file where the first column denotes the index
    (starting by 1) and the second column denotes the class label.
    """
    args = parser.parse_args()

    actions = glob.glob(os.path.join(args.action_folder, '*/'))
    actions = [path.split("/")[-2] for path in actions]
    actions = list(zip(range(1, len(actions) + 1), actions))
    print(actions)

    with open(args.csv_path, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for row in actions:
            writer.writerow(row)
    print('csv saved to %s' % args.csv_path)


if __name__ == '__main__':
    main()
