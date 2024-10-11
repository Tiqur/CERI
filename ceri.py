import cv2
import time
import numpy as np
from scipy.optimize import minimize
import rtree

class CERI:
    # Debugging
    image_index = 0

    def __init__(self):
        pass

    def optimize_thresholds(self, initial_thresholds, patience=5, min_improvement=0.01):
        self.best_score = float('inf')
        self.patience_counter = 0
        self.patience = patience
        self.min_improvement = min_improvement

        # Get all contours
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.precomputed_contours, self.precomputed_hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result = minimize(
            self.threshold_loss_function,
            initial_thresholds,
            method='Nelder-Mead',
            callback=self.optimization_callback,
            options={'maxiter': 1000}
        )
        return result.x

    def optimization_callback(self, xk):
        current_score = self.threshold_loss_function(xk)
        
        if current_score < self.best_score * (1 - self.min_improvement):
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"Stopping early due to lack of improvement after {self.patience} iterations.")
            return True

        return False

    def threshold_loss_function(self, thresholds):
        min_height_ratio, max_height_ratio = thresholds
        
        character_boxes = [
            cv2.boundingRect(c) for c in self.precomputed_contours
            if self.is_character(cv2.boundingRect(c), min_height_ratio, max_height_ratio)
        ]
        
        character_boxes.sort(key=lambda box: box[1])
        lines = self.find_connected_lines(character_boxes)
        score = self.calculate_line_score(lines)
        
        # Penalize very small thresholds
        small_threshold_penalty = 1000 * (0.005 / min_height_ratio) if min_height_ratio < 0.005 else 0
        
        return -score + small_threshold_penalty


    def find_connected_lines(self, boxes, max_horizontal_gap=8, max_vertical_deviation=5):
        lines = []
        current_line = []
        
        for box in boxes:
            if not current_line:
                current_line.append(box)
            else:
                prev_box = current_line[-1]
                if (self.horizontal_distance(prev_box, box) <= max_horizontal_gap and
                    self.vertical_deviation(prev_box, box) <= max_vertical_deviation):
                    current_line.append(box)
                else:
                    lines.append(current_line)
                    current_line = [box]
        
        if current_line:
            lines.append(current_line)
        
        return lines

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

    def calculate_line_score(self, lines):
        score = 0
        for line in lines:
            box_count = len(line)
            height_consistency = 1 / (1 + np.std([box[3] for box in line]))
            line_score = box_count * height_consistency
            score += line_score
        return score

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
        
        height, width, channels = self.image.shape

        initial_thresholds = [0.01, 0.03]
        optimized_thresholds = self.optimize_thresholds(initial_thresholds)
        optimized_thresholds = optimized_thresholds.flatten()
        print(optimized_thresholds)

        mint, maxt = optimized_thresholds
        threshold_avg_height_px = abs(mint*height-mint*maxt)
        horizontal_threshold = threshold_avg_height_px*0.8
        vertical_threshold = threshold_avg_height_px*0.6
        print(horizontal_threshold, vertical_threshold)

        self.identify_text_elements(optimized_thresholds, horizontal_threshold, vertical_threshold)

    def get_boxes_from_contours(self, contours):
        boxes = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
            #print(f"Contour {idx}: x={x}, y={y}, w={w}, h={h}")
        return boxes

    # Attempt to identify is that of a character
    def is_character(self, box, min_height_ratio=0.01, max_height_ratio=0.03):
        min_aspect_ratio = 0.2
        max_aspect_ratio = 5.0
        img_height, img_width, img_channels = self.image.shape
        x, y, w, h = box

        # Get box's height ratio compared to image height 
        box_height_ratio = h/img_height
        #print(box_height_ratio)
        
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
                if self.is_box_connected(current_string[-1], other_box, horizontal_threshold, vertical_threshold):
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

    # Return bounding boxes around words / sentences
    def identify_text_elements(self, thresholds, horizontal_threshold=10, vertical_threshold=4, min_area=0):
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
        character_boxes = [c for c in boxed_contours if self.is_character(c, *thresholds)]
        self.save_image_with_boxes(character_boxes)
        elapsed_time = time.time() - start_time
        print(f"Filter out non-characters: {elapsed_time:.4f} seconds")

        # Step 5: Filter to keep only innermost children (boxes without any children)
        start_time = time.time()
        filtered_boxes = self.filter_keep_innermost_children(character_boxes)
        self.save_image_with_boxes(filtered_boxes)
        elapsed_time = time.time() - start_time
        print(f"Filter to keep only innermost children: {elapsed_time:.4f} seconds")

        # Step 6: Merge characters into strings
        start_time = time.time()
        strings = self.merge_characters(filtered_boxes, horizontal_threshold, vertical_threshold)
        self.save_image_with_boxes(strings)
        elapsed_time = time.time() - start_time
        print(f"Merge characters into strings: {elapsed_time:.4f} seconds")








