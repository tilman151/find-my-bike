from PIL import Image


class UnifyingPad:
    def __init__(self, x: int, y: int):
        self.size = x, y

    def __call__(self, img: Image) -> Image:
        padded = Image.new("RGB", self.size)
        paste_pos = (self.size[0] - img.size[0]) // 2, (self.size[1] - img.size[1]) // 2
        padded.paste(img, paste_pos)

        return padded

    def __repr__(self) -> str:
        return f"UnifyingPad({self.size[0]}, {self.size[1]})"


class UnifyingResize:
    def __init__(self, max_size: int):
        self.max_size = max_size

    def __call__(self, img: Image.Image) -> Image.Image:
        h, w = img.size
        if h < w:
            resized = img.resize((int(self.max_size * h / w), self.max_size))
        else:
            resized = img.resize((self.max_size, int(self.max_size * w / h)))

        return resized

    def __repr__(self) -> str:
        return f"UnifyingPad({self.max_size})"
