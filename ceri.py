import cv2
import time
from rtree import index

class CERI:
    # Debugging
    image_index = 0

    def __init__(self):
        self.image = None
        self.precomputed_contours = None
        self.precomputed_hierarchy = None
        self.rtree = None 

    def build_rtree(self, bounding_boxes):
        self.rtree = index.Index()
        for idx, box in enumerate(bounding_boxes):
            x, y, w, h = box
            self.rtree.insert(idx, (x, y, x + w, y + h))

    def get_overlapping_ids(self, box, bounding_boxes):
        x, y, w, h = box
        return list(self.rtree.intersection((x, y, x + w, y + h)))

    def _collect_rgb_values(self, pixels, surrounding_pixels):
        """Helper function to collect RGB values and update counts."""
        for rgb in pixels.reshape(-1, 3):
            rgb_tuple = tuple(rgb)
            if rgb_tuple in surrounding_pixels:
                surrounding_pixels[rgb_tuple] += 1
            else:
                surrounding_pixels[rgb_tuple] = 1

    def get_characters(self, boxes, color_leniency, margin=1):
        characters = []

        for box in boxes:
            x, y, w, h = box

            # Dictionary to store RGB values and their counts
            surrounding_pixels = {}

            # Collect pixels from surrounding areas
            if y - margin >= 0:  # Above the box
                above_pixels = self.image[y - margin:y, x:x+w]
                self._collect_rgb_values(above_pixels, surrounding_pixels)

            if y + h + margin <= self.image.shape[0]:  # Below the box
                below_pixels = self.image[y + h:y + h + margin, x:x+w]
                self._collect_rgb_values(below_pixels, surrounding_pixels)

            if x - margin >= 0:  # Left of the box
                left_pixels = self.image[y:y+h, x - margin:x]
                self._collect_rgb_values(left_pixels, surrounding_pixels)

            if x + w + margin <= self.image.shape[1]:  # Right of the box
                right_pixels = self.image[y:y+h, x + w:x + w + margin]
                self._collect_rgb_values(right_pixels, surrounding_pixels)

            # Calculate the most common color and its ratio
            if surrounding_pixels:
                # Find the most common color and its count
                most_common_color, most_common_count = max(surrounding_pixels.items(), key=lambda item: item[1])
                ratio = most_common_count / (2*(w+h))

                # Check if the ratio meets the color leniency requirement
                if ratio >= color_leniency:
                    characters.append(box)

        return characters

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
        return boxes

    def merge_characters(self, boxes, horizontal_threshold, vertical_threshold):
        strings = []
        processed = set()
        height, width, channels = self.image.shape

        # Sort boxes by x-coordinate for efficient grouping
        sorted_boxes = sorted(boxes, key=lambda b: b[0])

        # Create an R-tree for efficient spatial querying
        rtree = index.Index()
        for idx, box in enumerate(sorted_boxes):
            x, y, w, h = box
            rtree.insert(idx, (x, y, x + w, y + h))

        # Iterate through each box
        for i, box in enumerate(sorted_boxes):
            if i in processed:
                continue

            # Start a new group with the current box
            current_string = [box]
            processed.add(i)

            # Get nearby boxes within the horizontal and vertical thresholds
            x, y, w, h = box
            nearby_box_ids = list(rtree.intersection((0, y-h//2-vertical_threshold, width, y+h//2+vertical_threshold)))

            # Sort nearby boxes by x-coordinate
            sorted_nearby_ids = sorted(nearby_box_ids, key=lambda b: sorted_boxes[b][0])

            # Check each nearby box for merging
            for j in sorted_nearby_ids:
                other_box = sorted_boxes[j]
                if j in processed:
                    continue

                # Merge if the boxes are connected
                if self.is_box_connected(current_string[-1], other_box, horizontal_threshold, vertical_threshold):
                    current_string.append(other_box)
                    processed.add(j)

            # Merge the boxes into a single bounding box
            min_x = min(rect[0] for rect in current_string)
            min_y = min(rect[1] for rect in current_string)
            max_x = max(rect[0] + rect[2] for rect in current_string)
            max_y = max(rect[1] + rect[3] for rect in current_string)
            merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)

            # Add the merged box to the result
            strings.append(merged_box)

        return strings

    def is_box_connected(self, box1, box2, max_horizontal_gap, max_vertical_deviation):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Horizontal distance between the boxes
        horizontal_distance = x2 - (x1 + w1)

        # Vertical deviation between the centers of the boxes
        center1 = y1 + h1 / 2
        center2 = y2 + h2 / 2
        vertical_deviation = abs(center1 - center2)
        
        return horizontal_distance <= max_horizontal_gap and vertical_deviation <= max_vertical_deviation

    def save_image_with_boxes(self, boxes):
        result_image = self.image.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        self.save_image(result_image)

    def remove_overlapping_boxes(self, boxes, threshold):
        # Sort boxes by area (high to low)
        sorted_boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        filtered_boxes = []

        def calculate_overlap(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Calculate the coordinates of the intersection rectangle
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            # Calculate the area of intersection
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate the area of the smaller box
            smaller_box_area = min(w1 * h1, w2 * h2)
            
            # Calculate the overlap ratio
            overlap_ratio = intersection_area / smaller_box_area
            
            return overlap_ratio

        for box in sorted_boxes:
            should_keep = True
            for kept_box in filtered_boxes:
                if calculate_overlap(box, kept_box) > threshold:  # Adjust this threshold as needed
                    should_keep = False
                    break
            
            if should_keep:
                filtered_boxes.append(box)

        return filtered_boxes

    #def filter_by_aspect_ratio(self, boxes, min_aspect_ratio, max_aspect_ratio):
        #return [box for box in boxes if min_aspect_ratio < (box[2] / box[3] if box[3] else 0) < max_aspect_ratio]

    def filter_by_area_ratio(self, boxes, max_area_ratio=0.2):
        image_area = self.image.shape[0] * self.image.shape[1]
        return [box for box in boxes if (box[2] * box[3]) / image_area <= max_area_ratio]

    def identify_text_elements(self, horizontal_threshold=10, vertical_threshold=4, min_area=0):
        # Step 1: Convert to grayscale
        start_time = time.time()
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.save_image(grayscale)
        elapsed_time = time.time() - start_time
        print(f"Convert to grayscale: {elapsed_time:.4f} seconds")

        # Step 2: Apply adaptive thresholding
        start_time = time.time()
        _ ,thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.save_image(thresh)
        elapsed_time = time.time() - start_time
        print(f"Apply adaptive thresholding: {elapsed_time:.4f} seconds")

        # Step 3: Find contours in the thresholded image
        start_time = time.time()
        self.precomputed_contours, self.precomputed_hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in self.precomputed_contours]
        self.save_image_with_boxes(bounding_boxes)
        elapsed_time = time.time() - start_time
        print(f"Find contours: {elapsed_time:.4f} seconds")

        # Step 4: Filter out boxes with area ratio larger than the threshold
        start_time = time.time()
        area_filtered_boxes = self.filter_by_area_ratio(bounding_boxes)
        self.save_image_with_boxes(area_filtered_boxes)
        print(f"Boxes after area ratio filtering: {len(area_filtered_boxes)}")
        elapsed_time = time.time() - start_time
        print(f"Filter out large area boxes: {elapsed_time:.4f} seconds")

        # Step 5: Filter out non-characters (surrounding pixels not same color)
        start_time = time.time()
        character_boxes = self.get_characters(area_filtered_boxes, color_leniency=0.8, margin=2)
        self.save_image_with_boxes(character_boxes)
        elapsed_time = time.time() - start_time
        print(f"Filter out non-characters: {elapsed_time:.4f} seconds")

        # Step 6: Build R-tree
        start_time = time.time()
        self.build_rtree(character_boxes)  # Build R-tree after finding contours
        self.save_image_with_boxes(character_boxes)
        elapsed_time = time.time() - start_time
        print(f"Build R-tree: {elapsed_time:.4f} seconds")

        # Step 7: Merge characters
        start_time = time.time()
        strings = self.merge_characters(character_boxes, horizontal_threshold, vertical_threshold)
        self.save_image_with_boxes(strings)
        elapsed_time = time.time() - start_time
        print(f"Merge characters into strings: {elapsed_time:.4f} seconds")

        # Step 8: Remove overlapping boxes
        start_time = time.time()
        non_overlapping = self.remove_overlapping_boxes(strings, 0.2)
        self.save_image_with_boxes(non_overlapping)
        elapsed_time = time.time() - start_time
        print(f"Removing all overlapping boxes: {elapsed_time:.4f} seconds")
