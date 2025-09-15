import cv2
import os

DATA_DIR = './EAC22008'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SYMBOLS_TO_COLLECT = ['1', '2', '3', '4']
dataset_size = 100

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Will collect data for these symbols: {SYMBOLS_TO_COLLECT}")
print("Press 'Q' to quit at any time.")

ROI_W, ROI_H = 192, 168

for j, current_data in enumerate(SYMBOLS_TO_COLLECT):
    class_dir = os.path.join(DATA_DIR, str(j+1))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {current_data} (Will be saved in folder: {j})')

    # --- Wait for key press ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate center ROI
        x1 = w // 2 - ROI_W // 2
        y1 = h // 2 - ROI_H // 2
        x2 = x1 + ROI_W
        y2 = y1 + ROI_H

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = f"Press '{current_data.lower()}' to start collecting for Class {current_data}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Data Collection', frame)

        key_pressed = cv2.waitKey(25)
        if key_pressed == ord(current_data.lower()):
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Starting in {countdown}...",
                            (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1000)
            break

        elif key_pressed == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # --- Image Capture ---
    counter = 1
    while counter <= dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        x1 = w // 2 - ROI_W // 2
        y1 = h // 2 - ROI_H // 2
        x2 = x1 + ROI_W
        y2 = y1 + ROI_H

        roi = frame[y1:y2, x1:x2]

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), roi)

        progress_text = f'Collecting for {current_data}: {counter}/{dataset_size}'
        cv2.putText(frame, progress_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)
        cv2.waitKey(50)

        counter += 1

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
