import cv2
import time
import numpy as np
from rtree import index

class CERI:
    # Debugging
    image_index = 0

    def __init__(self):
        self.image = None
        self.precomputed_contours = None
        self.precomputed_hierarchy = None
        self.rtree = index.Index()
        self.bounding_boxes = []

    def build_rtree(self):
        self.bounding_boxes = [cv2.boundingRect(c) for c in self.precomputed_contours]
        for idx, box in enumerate(self.bounding_boxes):
            x, y, w, h = box
            self.rtree.insert(idx, (x, y, x + w, y + h))

    def get_children(self, box):
        x, y, w, h = box
        overlapping = list(self.rtree.intersection((x, y, x + w, y + h)))
        return [idx for idx in overlapping if self.is_box_inside(self.bounding_boxes[idx], box)]

    def horizontal_distance(self, box1, box2):
        x1, _, w1, _ = box1
        x2, _, _, _ = box2
        return x2 - (x1 + w1)

    def vertical_deviation(self, box1, box2):
        _, y1, _, h1 = box1
        _, y2, _, h2 = box2
        center1 = y1 + h1 / 2
        center2 = y2 + h2 / 2
        return abs(center1 - center2)

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
            #print(f"Contour {idx}: x={x}, y={y}, w={w}, h={h}")
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

    def get_boxes_with_x_or_less_children(self, boxed_contours, max_child_count):
        # Create a dictionary to store the children of each box
        box_children_dict = {box: self.get_children(box) for box in boxed_contours}

        # Sort the boxed_contours by the number of children each box has, in descending order
        sorted_tuple_list = sorted(box_children_dict.items(), key=lambda item: len(item[1]), reverse=True)

        valid_boxes = []
        added_boxes = set()

        for box, children in sorted_tuple_list:
            # Check if the box already has too many children to be valid
            if len(children) > max_child_count:
                continue
            
            # Check if the box is already in added_boxes
            if box not in added_boxes:
                valid_boxes.append(box)  # Add the box to valid_boxes
                added_boxes.add(box)  # Mark the box as added

                # Only update added_boxes with unique children to prevent duplicates
                added_boxes.update(c for c in children if c not in added_boxes)

        return valid_boxes

    def filter_by_aspect_ratio(self, boxes, min_aspect_ratio, max_aspect_ratio):
        return [box for box in boxes if min_aspect_ratio < (box[2] / box[3] if box[3] else 0) < max_aspect_ratio]

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
        self.precomputed_contours, self.precomputed_hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.build_rtree()  # Build R-tree after finding contours
        self.save_image_with_boxes(self.bounding_boxes)
        elapsed_time = time.time() - start_time
        print(f"Find contours and build R-tree: {elapsed_time:.4f} seconds")

        # Step 4: Filter by aspect ratio 
        start_time = time.time()
        aspect_ratio_filtered_boxes = self.filter_by_aspect_ratio(self.bounding_boxes, min_aspect_ratio=0.2, max_aspect_ratio=3.0)
        self.save_image_with_boxes(aspect_ratio_filtered_boxes)
        elapsed_time = time.time() - start_time
        print(f"Filter out non-characters by aspect ratio: {elapsed_time:.4f} seconds")

        # Step 5: Filter out boxes with more than 1 child
        start_time = time.time()
        children_filtered_boxes = self.get_boxes_with_x_or_less_children(aspect_ratio_filtered_boxes, 3)
        self.save_image_with_boxes(children_filtered_boxes)
        elapsed_time = time.time() - start_time
        print(f"Filtered out boxes with more than 1 child: {elapsed_time:.4f} seconds")


        # Step 6: Merge characters into strings
        start_time = time.time()
        strings = self.merge_characters(children_filtered_boxes, horizontal_threshold, vertical_threshold)
        self.save_image_with_boxes(strings)
        elapsed_time = time.time() - start_time
        print(f"Merge characters into strings: {elapsed_time:.4f} seconds")








