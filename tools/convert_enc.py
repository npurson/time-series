import os
import os.path as osp


def walkdir(dir, suffix='.txt'):
    """Walks through subdirectories and
    returns a list of file paths.
    """
    paths = []
    for f in os.listdir(dir):
        path = osp.join(dir, f)
        if osp.isdir(path):
            paths += walkdir(path)
        elif path.endswith(suffix):
            paths.append(path)
    return paths


if __name__ == '__main__':
    paths = walkdir('data/nari0602')
    for path in paths:
        if 'FFCurrIn' not in path:
            continue
        with open(path, encoding='gbk') as f:
            c = f.read()
        with open(path, mode='w', encoding='utf-8') as f:
            f.write(c)
