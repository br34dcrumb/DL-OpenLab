import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from ultralytics import YOLO

import argparse

# --- Model and Class Configuration ---
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# COCO class names, focusing on vehicles
# Class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
vehicle_class_ids = [2, 3, 5, 7]

# Meter-to-pixel conversions
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 720

# Get current working directory
CWD_PATH = os.getcwd()

def readVideo(filename='drive.mp4'):
    return cv2.VideoCapture(os.path.join(CWD_PATH, filename))

def processImage(inpImage):
    # Apply HLS color filtering
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(hls, lower_white, upper_white)  # FIX: apply to HLS image
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask=mask)

    # Convert to grayscale and process
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    return inpImage, hls_result, gray, thresh, blur, canny

def perspectiveWarp(inpImage):
    img_size = (inpImage.shape[1], inpImage.shape[0])

    src = np.float32([[550, 475],
                      [256, 700],
                      [1200, 700],
                      [740, 475]])
    dst = np.float32([[256, 0],
                      [256, 720],
                      [1200, 720],
                      [1200, 0]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

    height, width = birdseye.shape[:2]
    birdseyeLeft = birdseye[:, :width // 2]
    birdseyeRight = birdseye[:, width // 2:]

    # Return the src points along with other values
    return birdseye, birdseyeLeft, birdseyeRight, minv, src, dst

def plotHistogram(inpImage):
    histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftxBase = np.argmax(histogram[:midpoint])
    rightxBase = np.argmax(histogram[midpoint:]) + midpoint
    return histogram, leftxBase, rightxBase

def slide_window_search(binary_warped, histogram):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if lefty.size == 0 or righty.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return ploty, left_fit, right_fit, left_fitx, right_fitx

def draw_lane_lines(original_image, warped_image, Minv, left_fitx, right_fitx, ploty):
    """
    Draws the lane lines and the area between them on the original image.
    """
    if left_fitx.size == 0 or right_fitx.size == 0:
        return original_image

    # Create a blank image to draw the lines on
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into a usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # The color is (0, 255, 255) which is Yellow in BGR format
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 255))

    cv2.polylines(color_warp, np.int_([pts]), isClosed=True, color=(0, 0, 255), thickness=5)
    
    # Warp the blank back to original image space using the inverse matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, new_warp, 0.3, 0)
    
    return result


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Set up argument parser
    parser = argparse.ArgumentParser(description="Lane and Vehicle Detection")
    parser.add_argument("--input", type=str, help="Path to the input video file. Leave blank for webcam.")
    parser.add_argument("--output", type=str, help="Path to save the output video file. Leave blank for real-time display.")
    args = parser.parse_args()

    # 2. Configure video input
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(0) # Default to webcam

    if not cap.isOpened():
        raise IOError("Cannot open video source")

    # 3. Configure video output (if --output is specified)
    writer = None
    if args.output:
        # Get video properties for the writer
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to {args.output}...")

    # 4. Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        birdView, birdViewL, birdViewR, minverse, src, dst = perspectiveWarp(frame)
        warped_height = birdView.shape[0]

        # cv2.polylines(frame, [np.int32(src)], isClosed=True, color=(255, 0, 0), thickness=2)
        # cv2.polylines(frame, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                if cls_id in vehicle_class_ids and confidence > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[cls_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    

        original_frame = np.copy(frame)
        _, _, _, thresh, _, _ = processImage(birdView)
        hist, leftBase, rightBase = plotHistogram(thresh)
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)
        result_image = draw_lane_lines(original_frame, thresh, minverse, left_fitx, right_fitx, ploty)
        # --- END OF PROCESSING LOGIC ---

        # 5. Handle output: either save or display
        if writer:
            writer.write(result_image)
        else:
            cv2.imshow("Lane and Car Detection", result_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 6. Release resources
    cap.release()
    if writer:
        writer.release()
        print("Video processing complete and file saved.")
    cv2.destroyAllWindows()
