import cv2
import numpy as np

# Global variables to store coordinates
area1 = []
area2 = []
current_area = 1  # Track whether we're defining area1 or area2

def mouse_callback(event, x, y, flags, param):
    global area1, area2, current_area

    # Left-click to add a point to the current area
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_area == 1 and len(area1) < 4:
            area1.append((x, y))
            print(f"Area1 Point {len(area1)}: ({x}, {y})")
        elif current_area == 2 and len(area2) < 4:
            area2.append((x, y))
            print(f"Area2 Point {len(area2)}: ({x}, {y})")

def main():
    global current_area

    # RTSP URL of your IP camera
    rtsp_url = "rtsp://admin:Nimda@2024@10.10.116.72:554/media/video1"

    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Create a window and set mouse callback
    cv2.namedWindow("Select Areas")
    cv2.setMouseCallback("Select Areas", mouse_callback)

    print("=== INSTRUCTIONS ===")
    print("1. Click 4 points to define AREA1 (entry/exit boundary).")
    print("2. Press 'n' to switch to AREA2.")
    print("3. Click 4 points to define AREA2.")
    print("4. Press 'q' to quit and save coordinates.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Resize the frame to match your main application
        frame = cv2.resize(frame, (1220, 720))  # Updated to (1220, 720)

        # Draw existing points and polygons
        if area1:
            for point in area1:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
            if len(area1) == 4:
                cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)

        if area2:
            for point in area2:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)
            if len(area2) == 4:
                cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

        # Show instructions on the frame
        cv2.putText(frame, f"Defining AREA{current_area}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Select Areas", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n') and current_area == 1 and len(area1) == 4:
            current_area = 2
            print("\nNow define AREA2:")
        elif key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Print the final coordinates
    print("\n=== FINAL COORDINATES ===")
    print("area1 =", area1)
    print("area2 =", area2)

if __name__ == "__main__":
    main()