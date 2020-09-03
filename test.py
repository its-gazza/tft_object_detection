from glob import glob


if __name__ == "__main__":
    filenames = glob('data/vid_1/raw/*.xml')
    for filename in filenames:
        with open(filename, 'r') as f:
            data = f.read()

        with open(filename, 'w') as f:
            data = data.replace('.jpg.jpg', '.jpg')
            f.write(data)
