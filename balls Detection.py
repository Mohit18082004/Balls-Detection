import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture(r"C:\Users\pranav\Dropbox\My PC (LAPTOP-666S6P8Q)\Downloads\WhatsApp Video 2024-04-02 at 7.56.13 PM.mp4")

# Initialize parameters for duplicate detection
prev_ball_positions = []
min_distance_threshold = 20

# Initialize a counter for total number of balls
total_balls = 0

# Loop through each frame of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if no frame is read
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Detect circles in the blurred frame
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=50)

    # If circles are found, draw a bounding box around each ball and label them
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Check for duplicate detection
            is_duplicate = False
            for (prev_x, prev_y, _) in prev_ball_positions:
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if distance < min_distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Draw circle and label only if not a duplicate
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.putText(frame, f'Ball {total_balls + 1}', (x - 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                total_balls += 1
                # Update previous ball positions
                prev_ball_positions.append((x, y, r))

    # Display the frame with detected balls
    cv2.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the total number of balls detected in the video
print("Total number of balls detected in the video:", total_balls)

