### Marcus Hock, Paul Salame, Rebekah Haysley, Cameron Casby, Andrew Oliver
### Baller Sparki
### Obstacle Identification and mapping

import numpy as np
import cv2 as cv
import colorsys
import matplotlib.pyplot as plt


# Function that takes an image in and bgr_color code as an array
def mask_and_remove(in_image, color):
    color_8bit = np.uint8([[np.array(color)]]) #Format color
    hsv_color = cv.cvtColor(color_8bit,cv.COLOR_BGR2HSV)[0][0] #Convert to HSL for easier filtering

    tol = 7 # What range of colors will be allowed

    hue_lower = hsv_color[0]-tol #Set lower bound for hue

    # In case hue is above 180 which is the bounds for open CV we need to account for this
    if hue_lower < 0:
        hue_lower = 180-hue_lower

    # In case hue is below 0, account for this
    hue_upper = hsv_color[0]+tol # Set upper bound for hue
    if hue_upper> 180:
        hue_upper = hue_upper - 180

    # Create bounds for cv in range function
    lower_bound = np.array([hue_lower,50,50])
    upper_bound = np.array([hue_upper,255,255])

    # Convert to HSL for easier manilulation
    hsv_image = cv.cvtColor(in_image, cv.COLOR_BGR2HSV) # Convert image to HSV
    image_mask = cv.inRange(hsv_image, lower_bound, upper_bound) # Create Mask

    isolated_im = cv.bitwise_and(in_image,in_image, mask= image_mask) # Combine mask and image

    # Create new image with isolated image and white background
    white_bkgd = isolated_im
    white_bkgd[np.where((isolated_im==[0,0,0]).all(axis=2))] = [255,255,255]  # Combine images

    return(white_bkgd) #Function resturns an image that is color filtered, and includes a white background

# A function that performs blob detection and returns the locations of the blobs
def blob_locations(clean_im): # Takes an image that has been color isolate
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Set thresholds
    params.minThreshold = 5
    params.maxThreshold = 3000

    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 1000000

    # Create detector paramters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(clean_im)

    # Create an openCV image that circles the blobs
    im_with_keypoints = cv.drawKeypoints(clean_im, keypoints, np.array([]), (0,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv.imshow("Keypoints", im_with_keypoints)
    cv.waitKey(0)

    # Create list of blob locations
    locations = []
    for point in keypoints:
        locations.append([point.pt[0],point.pt[1]])
    return(locations) #Reutns list with blob locations

# Takes an input of the original image and the coordinates of the cornersself.
# Produces an image that has been transformed so the pixel and real space match
def transform_image(corners,original_image):
    #Define new corners
    first_corner = np.array([0,0])
    second_corner = np.array([400,0])
    third_corner = np.array([0,400])
    fourth_corner = np.array([400,400])
    new_corners = np.array([first_corner, second_corner, third_corner, fourth_corner])

    ordered = np.zeros((4,2))

    # Identify which blobs are which coordinates
    i = 0
    for s in new_corners:
        dists= []
        for point in corners:
            dists.append(np.linalg.norm(np.array(s)-np.array(point)))
        ind = np.argmin(dists)

        ordered[i]= np.array(corners[ind])
        i+=1

    # Create a transformation matrix
    transform_matrix = cv.getPerspectiveTransform(np.float32(ordered),np.float32(new_corners))

    # Transform image
    new_image = cv.warpPerspective(original_image,transform_matrix,(400,400))
    return(new_image) #Return new image

######################
# Main script region #
######################
map_image = cv.imread("Final_Map.jpg") # Load image

# Define green and orange colors
bgr_green = [50,95,20]
bgr_orange = [27,83,191]

# Resize image
dims = np.shape(map_image)
map_image = cv.resize(map_image,(int(dims[1]/10),int(dims[0]/10)))

# Produce an image that includes the only the green corner images
cleaned_image = mask_and_remove(map_image,bgr_green) # Can be used for blob detection

# Show the image after color filtering for green
cv.imshow("Cleaned",cleaned_image)
cv.waitKey(0)

# Define corners of map
corners = blob_locations(cleaned_image)

# Transorm the original image based on the corners of the map
new_image = transform_image(corners, map_image)

# Show image after transformation
cv.imshow('Transformed',new_image)
cv.waitKey(0)

# Filter the transormed image for the color orange
cleaned_image_2 = mask_and_remove(new_image,bgr_orange)

# Show image after orange color selection
cv.imshow('Transformed',cleaned_image_2)
cv.waitKey(0)

# Identify orange blobs, and therefore obstacles
obstacles = blob_locations(cleaned_image_2)

# Converts pixel locations to the coordinate locations
points = np.array(obstacles)
x_points = points[:,0]
y_points = 400-points[:,1]

# Create a plot to show the map in matplotlib
fig, ax = plt.subplots(1,1,figsize = (8,8))
for i in range(7):
    ax.plot([0,6],[i,i],'k')

for i in range(7):
    ax.plot([i,i],[0,6],'k')

ax.scatter(x_points/400*6,y_points/400*6)
ax.set_xlabel('i', fontsize = 20)
ax.set_ylabel('j', fontsize = 20)
plt.show()

i = np.floor(x_points/400*6)
j = np.floor(y_points/400*6)
print("i and j coordinates of obstacles to enter into Sparki's system: ")
print('i:', i)
print('j:',j)
