import glob
import imageio

imageio.plugins.freeimage.download()

anim_file = 'output.gif'

filenames = glob.glob('tmp/img_*.jpg')
filenames = sorted(filenames)
last = -1
images = []
for i in range(368):
    filepath = f"./tmp/img_{i}.jpg"
    image = imageio.imread(filepath)
    images.append(image)

imageio.mimsave(anim_file, images, 'GIF-FI', fps=10)