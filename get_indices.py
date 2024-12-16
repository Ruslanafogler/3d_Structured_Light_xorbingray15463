import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from triangulate import read_img
import numpy as np
import glob
import os

def select_points(img_path):
    """
    Displays an image and allows the user to select two points.
    Returns the coordinates of the selected points.

    Args:
        img_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the coordinates of the two selected points.
    """

    img = mpimg.imread(img_path)

    fig, ax = plt.subplots()
    ax.imshow(img)

    points = []

    def onclick(event):
        x, y = int(event.xdata), int(event.ydata)
        points.append(np.array([y, x]))
        rect = patches.Rectangle((x, y), 2048, 2048, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return points

if __name__ == "__main__":

    images = glob.glob(os.path.join("./data", "binary", "*"+'.JPG'))
    print("paths", images)

    offsets = []
        # Example usage:

    for i, p in enumerate(images):
        print("i, p", i, p)

        points = select_points(p)
        print("point is", points)

        offsets.append(points)

        # Extract pixel values from the selected points
        x1, y1 = points[0]

        # Continue with your program using the extracted pixel values
    offsets = np.vstack(offsets)
    print("offsets are", offsets)
    np.save("offsets.npy", offsets)