import cv2
import time
import numpy as np

class CERI:
    # Debugging
    image_index = 0

    def __init__(self):
        pass

    def save_image(self, img, name=None):
        if name is None:
            name = self.image_index

        cv2.imwrite(f'{name}.png', img)
        print(f'Saved {name}.png')
        self.image_index += 1

    def detect(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Failed to load the image.")

        self.identify_text_elements()

    def get_boxes_from_contours(self, contours):
        boxes = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
            print(f"Contour {idx}: x={x}, y={y}, w={w}, h={h}")
        return boxes

    # Attempt to identify is that of a character
    def is_character(self, box, min_aspect_ratio=0.2, max_aspect_ratio=5.0, min_height_ratio=0.01, max_height_ratio=0.03):
        img_height, img_width, img_channels = self.image.shape
        x, y, w, h = box

        # Get box's height ratio compared to image height 
        box_height_ratio = h/img_height
        print(box_height_ratio)
        
        # Get aspect ratio and area
        aspect_ratio = w/h if h else 0

        return (
            min_aspect_ratio < aspect_ratio < max_aspect_ratio and
            min_height_ratio < box_height_ratio < max_height_ratio
        )

    def merge_characters(self, boxes, horizontal_threshold, vertical_threshold):
        strings = []
        processed = set()

        # Sort boxes by x-coordinate for efficient grouping
        boxes.sort(key=lambda b: b[0])

        for i, box in enumerate(boxes):
            if i in processed:
                continue

            # Start a new group for the group
            current_string = [box]
            processed.add(i)

            # Check for other boxes to group them with the current string
            for j, other_box in enumerate(boxes):
                if j in processed:
                    continue

                # Horizontal merging
                if self.character_within_threshold(current_string[-1], other_box, horizontal_threshold, vertical_threshold):
                    current_string.append(other_box)
                    processed.add(j)

            min_x = min(rect[0] for rect in current_string)
            min_y = min(rect[1] for rect in current_string)
            max_x = max(rect[0] + rect[2] for rect in current_string)
            max_y = max(rect[1] + rect[3] for rect in current_string)
            merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)

            # Add the merged box to the result
            strings.append(merged_box)

        return strings

    def character_within_threshold(self, box1, box2, horizontal_threshold, vertical_threshold):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Horizontal distance between the boxes
        horizontal_distance = x2 - (x1 + w1)

        top = abs((y1 + h1) - (y2 + h2))
        bottom = abs(y1 - y2)
        middle = abs((y1+h1//2) - (y2+h2//2))
        
        return horizontal_distance <= horizontal_threshold and (
            top <= vertical_threshold or
            bottom <= vertical_threshold or
            middle <= vertical_threshold
        )

    def save_image_with_boxes(self, boxes):
        result_image = self.image.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        self.save_image(result_image)

    def is_box_inside(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2

    def count_border_pixels(self, box, thresh):
        x, y, w, h = box
        # Create a mask for the border of the box
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, 1)
        
        # Count non-zero pixels in the border
        return cv2.countNonZero(cv2.bitwise_and(thresh, mask))

    def filter_keep_character_contours(self, boxes, thresh):
        filtered_boxes = []
        for i, box in enumerate(boxes):
            keep_box = True
            for j, other_box in enumerate(boxes):
                if i != j:
                    if self.is_box_inside(box, other_box):
                        # If this box is inside another box, compare border pixels
                        if self.count_border_pixels(box, thresh) < self.count_border_pixels(other_box, thresh):
                            keep_box = False
                            break
                    elif self.is_box_inside(other_box, box):
                        # If this box contains another box, compare border pixels
                        if self.count_border_pixels(other_box, thresh) > self.count_border_pixels(box, thresh):
                            keep_box = False
                            break
            if keep_box:
                filtered_boxes.append(box)
        return filtered_boxes

    # Return bounding boxes around words / sentences
    def identify_text_elements(self, horizontal_threshold=10, vertical_threshold=4, min_area=0):
        # Step 1: Convert to grayscale
        start_time = time.time()
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.save_image(grayscale)
        elapsed_time = time.time() - start_time
        print(f"Convert to grayscale: {elapsed_time:.4f} seconds")

        # Step 2: Apply adaptive thresholding
        start_time = time.time()
        thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        self.save_image(thresh)
        elapsed_time = time.time() - start_time
        print(f"Apply adaptive thresholding: {elapsed_time:.4f} seconds")

        # Step 3: Find contours in the thresholded image
        start_time = time.time()
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxed_contours = [b for b in self.get_boxes_from_contours(contours) if (b[2] * b[3]) > min_area]
        self.save_image_with_boxes(boxed_contours)
        elapsed_time = time.time() - start_time
        print(f"Find contours in the thresholded image: {elapsed_time:.4f} seconds")

        # Step 4: Attempt to filter out non-characters
        start_time = time.time()
        character_boxes = [c for c in boxed_contours if self.is_character(c)]
        self.save_image_with_boxes(character_boxes)
        elapsed_time = time.time() - start_time
        print(f"Filter out non-characters: {elapsed_time:.4f} seconds")

        # Step 5: Filter to keep only innermost children (boxes without any children)
        start_time = time.time()
        filtered_boxes = self.filter_keep_character_contours(character_boxes, thresh)
        self.save_image_with_boxes(filtered_boxes)
        elapsed_time = time.time() - start_time
        print(f"Filter to keep only innermost children: {elapsed_time:.4f} seconds")

        # Step 6: Merge characters into strings
        start_time = time.time()
        strings = self.merge_characters(filtered_boxes, horizontal_threshold, vertical_threshold)
        self.save_image_with_boxes(strings)
        elapsed_time = time.time() - start_time
        print(f"Merge characters into strings: {elapsed_time:.4f} seconds")








