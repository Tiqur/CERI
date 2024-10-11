import cv2

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

    def filter_keep_innermost_children(self, boxes):
        innermost_children = []
        for i, box in enumerate(boxes):
            has_children = False
            for j, other_box in enumerate(boxes):
                if i != j and self.is_box_inside(other_box, box):
                    has_children = True
                    break
            if not has_children:
                innermost_children.append(box)
        return innermost_children

    # Return bounding boxes around words / sentences
    def identify_text_elements(self, horizontal_threshold=10, vertical_threshold=4, min_area=0):
        # Step 1: Convert to grayscale
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.save_image(grayscale)

        # Step 2: Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        self.save_image(thresh)

        # Step 3: Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxed_contours = [b for b in self.get_boxes_from_contours(contours) if (b[2]*b[3]) > min_area] # Filter out boxes with area smaller than min_area

        # Step 4: Attempt to filter out non-characters 
        character_boxes = [c for c in boxed_contours if self.is_character(c)]
        self.save_image_with_boxes(character_boxes)

        # Step 5: Filter to keep only innermost children (boxes without any children)
        filtered_boxes = self.filter_keep_innermost_children(character_boxes)
        self.save_image_with_boxes(filtered_boxes)

        # Step 6: Merge characters into strings
        strings = self.merge_characters(filtered_boxes, horizontal_threshold, vertical_threshold)
        self.save_image_with_boxes(strings)











