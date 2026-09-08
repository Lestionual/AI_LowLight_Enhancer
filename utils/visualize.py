import matplotlib.pyplot as plt


def show_image(tensor):

    image = tensor.permute(1, 2, 0).cpu().numpy()

    plt.imshow(image)

    plt.axis("off")

    plt.show()