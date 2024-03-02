import cv2

# Initialize list to hold coordinates
clicked_points = []

def click_event(event, x, y, flags, params):
    # if the left mouse button was clicked, record the starting (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point: ({x}, {y})")
        clicked_points.append((x, y))

# Load the image
image_path = 'inputs/2d_field.png'  # replace with your image path
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    print('Could not open or find the image:', image_path)
    exit(0)

# Create a window
cv2.namedWindow('image')

# Set a callback function for any mouse event on the window
cv2.setMouseCallback('image', click_event)

while True:
    # Display the image
    cv2.imshow('image', image)
    
    # Wait for the 's' key to be pressed to exit
    if cv2.waitKey(20) & 0xFF == ord('s'):
        print("Coordinates saved:", clicked_points)
        break

# Close the window
cv2.destroyAllWindows()