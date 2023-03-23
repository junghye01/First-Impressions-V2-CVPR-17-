from torchvision.transforms import Resize, Compose, Normalize


class Chalearn_VideoTransform:
    def __init__(self):
        self.transform = Compose(
            [Resize((112, 112), antialias=True), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __call__(self, x):
        return self.transform(x)
