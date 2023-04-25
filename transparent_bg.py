import cv2
bg_image = cv2.imread('./transparent.png')

# Convert from BGR to RGBA
bg_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

# Create a mask by thresholding
_, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# Set the alpha channel of the image to zero where the mask is zero
bg_image[:, :, 3] = mask[:, :, 0]

# Save the resulting image as a PNG file to preserve transparency
cv2.imwrite('try.png', bg_image)

bg_image = cv2.resize(bg_image, (width, height))