import os
import tarfile
from glob import glob

def compress_dir(input: str, target: str):
    os.makedirs(target, exist_ok=True)
    tar = tarfile.open(target + ".tar.xz", mode='w:xz')

    sub_files = glob(f'{input}/**', recursive=True)

    for f in sub_files:
        if os.path.isdir(f): continue

        name = f.replace(f'{input}/', '')
        tar.add(f, arcname=name)

    tar.close()

def uncompress_dir(input: str, target: str):
    os.makedirs(target, exist_ok=True)
    tar = tarfile.open(input + '.tar.xz', mode='r:xz')
    tar.extractall(target)
    tar.close()
