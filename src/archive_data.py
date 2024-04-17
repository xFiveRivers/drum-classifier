"""
Data Archiver

Usage:
    archive_data.py (-h | --help)
    archive_data.py <action> <target>

Options:
    -h --help           Show help screen.
    <action>            Flag to compress or extract data.
    <target>            Flag to select raw, clean, or both types of data.
"""


from utility import FileArchiver
from docopt import docopt


def main():
    args = docopt(__doc__)

    filearchiver = FileArchiver()

    if args['<action>'] == 'compress':
        filearchiver.compress_data(args['<target>'])
    elif args['<action>'] == 'extract':
        filearchiver.extract_data(args['<target>'])
    else:
        print('Please specify if you want to compress or extract data.')


if __name__ == '__main__':
    main()