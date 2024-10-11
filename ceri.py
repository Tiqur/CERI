import cv2

class CERI:
    def __init__(self):
        pass

    def save_image(self, img, name):
        cv2.imwrite(f'{name}.png', img)
        print(f'Saved {name}.png')

    def detect(self, image_path):
        self.image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Failed to load the image.")

        self.save_image(self.image, "1")

