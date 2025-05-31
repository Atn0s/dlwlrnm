import os
import time

import numpy as np
import pycuda.autoinit  # noqa F401: Initializes CUDA context
import pycuda.driver as cuda
import torch
import tensorrt as trt
from PIL import Image
import cv2  # For drawing on image

# Assuming rfdetr.datasets.transforms is available and configured as needed
# If not, ensure the placeholder or torchvision equivalent is correctly set up.
try:
    import rfdetr.datasets.transforms as RFT

    print("Using rfdetr.datasets.transforms.")
except ImportError:
    print("FATAL: rfdetr.datasets.transforms not found. Please install or provide an alternative.")
    exit()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference:
    def __init__(self, engine_path):
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        if not self.engine:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # --- Known tensor names and shapes from your log ---
        self.input_name = "input"
        self.input_shape = (1, 3, 560, 560)  # Batch, Channel, Height, Width
        self.input_dtype = np.float32

        self.output_dets_name = "dets"
        self.output_dets_shape = (1, 300, 4)  # Batch, NumDetections, BoxCoords (e.g., x1,y1,x2,y2 or cx,cy,w,h)

        self.output_labels_name = "labels"
        # From your log, labels seem to be (1, 300, 3)
        # Let's assume [class_id, score, ?]
        self.output_labels_shape = (1, 300, 3)
        self.output_dtype = np.float32  # Both outputs are float32 based on log

        # Allocate host (CPU) and device (GPU) buffers
        # Input
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=self.input_dtype)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.d_input))

        # Output 'dets'
        self.h_output_dets = cuda.pagelocked_empty(trt.volume(self.output_dets_shape), dtype=self.output_dtype)
        self.d_output_dets = cuda.mem_alloc(self.h_output_dets.nbytes)
        self.context.set_tensor_address(self.output_dets_name, int(self.d_output_dets))

        # Output 'labels'
        self.h_output_labels = cuda.pagelocked_empty(trt.volume(self.output_labels_shape), dtype=self.output_dtype)
        self.d_output_labels = cuda.mem_alloc(self.h_output_labels.nbytes)
        self.context.set_tensor_address(self.output_labels_name, int(self.d_output_labels))

        self.stream = cuda.Stream()
        print(f"TensorRT engine '{engine_path}' loaded and buffers prepared.")

    def preprocess_image(self, image_pil, target_transforms):
        img_transformed, _ = target_transforms(image_pil, None)
        input_np = img_transformed.numpy().astype(self.input_dtype)
        if input_np.ndim == 3:
            input_np = np.expand_dims(input_np, axis=0)
        if input_np.shape != self.input_shape:
            raise ValueError(f"Preprocessed image shape {input_np.shape} "
                             f"does not match model's expected input shape {self.input_shape}.")
        return np.ascontiguousarray(input_np)

    def infer(self, preprocessed_image):
        np.copyto(self.h_input, preprocessed_image.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output_dets, self.d_output_dets, self.stream)
        cuda.memcpy_dtoh_async(self.h_output_labels, self.d_output_labels, self.stream)
        self.stream.synchronize()

        output_dets_data = self.h_output_dets.reshape(self.output_dets_shape)
        output_labels_data = self.h_output_labels.reshape(self.output_labels_shape)

        return {self.output_dets_name: output_dets_data, self.output_labels_name: output_labels_data}

    def postprocess(self, raw_outputs, confidence_threshold=0.5, top_k=5):
        pred_boxes_np = raw_outputs[self.output_dets_name]
        # logits_np now correctly understood as raw class logits for each detection proposal
        class_logits_np = raw_outputs[self.output_labels_name]  # Shape (1, 300, num_classes) where num_classes is 3

        print(f"Debug: class_logits_np shape: {class_logits_np.shape}")
        if class_logits_np.shape[0] > 0 and class_logits_np.shape[1] > 0:
            print(f"Debug: First proposal's class logits (first 5 proposals): {class_logits_np[0, :5, :]}")

        # Convert to torch tensor and apply sigmoid to get probabilities for each class
        # Assuming batch size is 1, so squeeze it out.
        # class_logits_np shape is (1, 300, 3)
        class_probs = torch.sigmoid(torch.from_numpy(class_logits_np).squeeze(0))  # Shape (300, 3)

        # For each detection proposal (300 of them), find the class with the highest probability
        # max_scores will be the probability of the most likely class for each proposal
        # pred_class_ids will be the index of that class (0, 1, or 2)
        max_scores, pred_class_ids = torch.max(class_probs, dim=1)  # Max over the num_classes dimension (dim=1)
        # max_scores shape (300,), pred_class_ids shape (300,)

        all_boxes = torch.from_numpy(pred_boxes_np).squeeze(0)  # Shape (300, 4)

        # Filter by confidence threshold
        confidence_mask = max_scores > confidence_threshold

        filtered_scores = max_scores[confidence_mask]
        filtered_labels = pred_class_ids[confidence_mask]
        filtered_boxes = all_boxes[confidence_mask]

        if filtered_scores.numel() == 0:
            print("No detections meet the confidence threshold after sigmoid and max.")
            return [], [], []

        num_to_keep = min(top_k, filtered_scores.numel())
        top_indices = filtered_scores.argsort(descending=True)[:num_to_keep]

        top_k_scores = filtered_scores[top_indices].tolist()
        top_k_labels = filtered_labels[top_indices].tolist()
        top_k_boxes = filtered_boxes[top_indices].tolist()

        return top_k_labels, top_k_scores, top_k_boxes

    def detect_single_image(self, image_path, target_transforms, confidence_threshold=0.5, top_k=5):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img_pil = Image.open(image_path).convert("RGB")

        preprocessed_image = self.preprocess_image(img_pil, target_transforms)
        raw_outputs = self.infer(preprocessed_image)
        return self.postprocess(raw_outputs, confidence_threshold, top_k)

    def cleanup(self):
        if self.d_input: self.d_input.free()
        if self.d_output_dets: self.d_output_dets.free()
        if self.d_output_labels: self.d_output_labels.free()
        print("CUDA device memory allocations cleaned up.")


def draw_detections(image_cv2, labels, scores, boxes, class_names=None, line_thickness=2):
    """Draws detection boxes, labels, and scores on an OpenCV image."""
    display_image = image_cv2.copy()
    for label_id, score, box_coords in zip(labels, scores, boxes):  # Renamed box to box_coords for clarity
        # box_coords is expected to be [x_min, y_min, x_max, y_max] in pixel coordinates
        x1, y1, x2, y2 = map(int, box_coords)  # This correctly converts to int for drawing

        # Ensure coordinates are within image bounds (optional, but good practice)
        h, w = display_image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # Only draw if the box has a valid area
        if x2 > x1 and y2 > y1:
            class_name = str(label_id)
            if class_names and label_id < len(class_names):
                class_name = class_names[label_id]

            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

            text = f"{class_name}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)

            # Adjust text position if it goes off-screen (simple adjustment)
            text_y_pos = y1 - baseline // 2
            bg_y1_pos = y1 - text_height - baseline
            if bg_y1_pos < 0:  # If text background goes above image top
                bg_y1_pos = y1 + baseline
                text_y_pos = y1 + text_height + baseline // 2

            cv2.rectangle(display_image, (x1, bg_y1_pos), (x1 + text_width, bg_y1_pos + text_height + baseline),
                          (0, 255, 0), -1)
            cv2.putText(display_image, text, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            print(f"Skipping drawing invalid box: {[x1, y1, x2, y2]}")

    return display_image


if __name__ == "__main__":
    ENGINE_PATH = "./output/inference_model.sim.engine"
    IMAGE_PATH = "./pictures/07.jpg"
    OUTPUT_IMAGE_PATH = "./pictures/07_detected.jpg"  # Path to save image with detections
    CONFIDENCE_THRESHOLD = 0.5
    TOP_K_RESULTS = 5  # Show top K results

    # --- Define Transforms (Must match model's training/export) ---
    # Using 560 as per your successful log with SquareResize([560])
    # The actual resize behavior of SquareResize([N]) depends on its implementation.
    # It might resize the shortest side to N and keep aspect ratio, or make it NxN.
    # This needs to align with the engine's (1,3,560,560) input.
    current_transforms = RFT.Compose([
        RFT.SquareResize([560]),  # This should result in a 560x560 image for the model
        RFT.ToTensor(),
        RFT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Optional: Define class names if you have them (index should match class_id)
    CLASS_NAMES = ["unknown", "blue", "red"]  # Example: replace with your actual class names

    if not os.path.exists(ENGINE_PATH):
        print(f"CRITICAL ERROR: TensorRT engine file '{ENGINE_PATH}' not found.")
        exit()
    if not os.path.exists(IMAGE_PATH):
        print(f"CRITICAL ERROR: Image file '{IMAGE_PATH}' not found.")
        exit()

    trt_model = None
    try:
        trt_model = TRTInference(ENGINE_PATH)

        start_time = time.time()
        detected_labels, detected_scores, detected_boxes = trt_model.detect_single_image(
            IMAGE_PATH,
            target_transforms=current_transforms,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            top_k=TOP_K_RESULTS
        )
        end_time = time.time()

        print(f"\n--- Detection Results on Image: {IMAGE_PATH} ---")
        if detected_labels:
            for i, (label, score, box) in enumerate(zip(detected_labels, detected_scores, detected_boxes)):
                class_name_str = CLASS_NAMES[label] if CLASS_NAMES and label < len(CLASS_NAMES) else f"ID {label}"
                box_str = [f"{coord:.2f}" for coord in box]
                print(f"  Detection {i + 1}: Class: {class_name_str}, Score: {score:.4f}, Box: [{', '.join(box_str)}]")

            # Load image with OpenCV to draw detections
            original_image_cv2 = cv2.imread(IMAGE_PATH)
            if original_image_cv2 is None:
                print(f"Error: Could not read image {IMAGE_PATH} with OpenCV.")
            else:
                h_orig, w_orig = original_image_cv2.shape[:2]

                pixel_boxes_for_drawing = []
                for box_norm in detected_boxes:  # box_norm is [cx_n, cy_n, w_n, h_n]
                    cx_n, cy_n, w_n, h_n = box_norm

                    # Convert normalized [cx, cy, w, h] to pixel [x_min, y_min, x_max, y_max]
                    center_x_px = cx_n * w_orig
                    center_y_px = cy_n * h_orig
                    width_px = w_n * w_orig
                    height_px = h_n * h_orig

                    x_min = center_x_px - (width_px / 2)
                    y_min = center_y_px - (height_px / 2)
                    x_max = center_x_px + (width_px / 2)
                    y_max = center_y_px + (height_px / 2)

                    pixel_boxes_for_drawing.append([x_min, y_min, x_max, y_max])

                image_with_detections = draw_detections(original_image_cv2, detected_labels, detected_scores,
                                                        pixel_boxes_for_drawing, CLASS_NAMES)

                cv2.imshow("Detections", image_with_detections)
                print(f"\nDisplaying image with detections. Press any key to close window and save.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imwrite(OUTPUT_IMAGE_PATH, image_with_detections)
                print(f"Image with detections saved to: {OUTPUT_IMAGE_PATH}")

        else:
            print("  No objects detected or met the confidence/top_k criteria.")

        print(f"\nTotal inference time (including pre/post): {end_time - start_time:.4f} seconds")

    except RuntimeError as e:
        print(f"ERROR - RuntimeError: {e}")
    except FileNotFoundError as e:
        print(f"ERROR - FileNotFoundError: {e}")
    except Exception as e:
        print(f"ERROR - An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if trt_model:
            trt_model.cleanup()

    print("Script execution finished.")