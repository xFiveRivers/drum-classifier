import os
import zipfile


class FileArchiver():
    def __init__(self):
        self.data_list = ['data/raw', 'data/clean']
        self.archive_list = [x + '.zip' for x in self.data_list]


    def extract_data(self, target: str = 'both'):
        if target == 'raw':
            self._extract(self.archive_list[0])
        elif target == 'clean':
            self._extract(self.archive_list[1])
        else:
            for zip in self.archive_list:
                self._extract(zip)
    

    def compress_data(self, target: str = 'both'):
        if target == 'raw':
            self._compress(self.archive_list[0], self.data_list[0])
        elif target == 'clean':
            self._compress(self.archive_list[1], self.data_list[1])
        else:
            for archive, dir in zip(self.archive_list, self.data_list):
                self._compress(archive, dir)


    def _extract(self, archive_name: str):
        print(f'Extracting {archive_name}...')

        with zipfile.ZipFile(archive_name, mode="r") as archive:
            archive.extractall()

        print('Extraction complete!')
    

    def _compress(self, archive_name: str, dir: str):
        print(f'Compressing {dir} to {archive_name}')

        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as archive:
            for root, _, files in os.walk(dir):
                for file in files:
                    archive.write(
                        os.path.join(root, file),
                        os.path.relpath(
                            os.path.join(root, file), 
                            os.path.join(dir, '..')
                        )
                    )
        archive.close()

        print('Compression complete!')