import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

def calculate_match_percentage(matches):
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    match_percentage = (len(good_matches) / len(matches)) * 100
    return match_percentage, good_matches

# Read the query image as query_img
# and train image This query image
# is what you need to find in the train image
# Save it in the same directory
# with the name image.jpg
query_img = cv2.imread('qqqq.png')
train_img = cv2.imread('product.png')

# Check if the images were loaded successfully
if query_img is None:
    print("Error: Could not load query image.")
elif train_img is None:
    print("Error: Could not load train image.")
else:
    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector algorithm
    sift = cv2.SIFT_create()

    # Now detect the keypoints and compute
    # the descriptors for the query image
    # and train image
    queryKeypoints, queryDescriptors = sift.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = sift.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

    # Apply ratio test
    match_percentage, good_matches = calculate_match_percentage(matches)

    print(f"Match Percentage: {match_percentage:.2f}%")

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Image Matching")

    # Display the match percentage on the Tkinter window
    match_percentage_out_of_100 = match_percentage / 10.0
    label = tk.Label(root, text=f"Match Percentage: {match_percentage_out_of_100:.2%}")
    label.pack()

    # Draw the matches to the final image
    final_img = cv2.drawMatchesKnn(query_img, queryKeypoints,
                                   train_img, trainKeypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Convert the final image to RGB format for displaying in Tkinter
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(final_img_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Create a Tkinter label to display the final image
    label_img = tk.Label(root, image=img_tk)
    label_img.pack()

    # Run the Tkinter main loop
    root.mainloop()
