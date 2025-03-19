# q1
# import cv2
# img=cv2.imread('Cat_November_2010-1a.jpg ')
# resized=cv2.resize(img,(300,300))
# cv2.imshow('image',resized)
# (h, w) = img.shape[:2]
# center = (w // 2, h // 2)
# rotation_matrix = cv2.getRotationMatrix2D(center, 150, 1.0)  
# rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))
# cv2.imshow('Rotated Image', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# q2.
# import cv2
# img=cv2.imread('Cat_November_2010-1a.jpg ')
# cv2.rectangle(img, (200, 400), (200 + 230, 400 + 250), (0, 0, 255), 2)  # Red color (BGR), thickness=2

#     # Write "YOUR NAME" below the rectangle in yellow
# text = "YOUR NAME"
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 0.8
# font_thickness = 2
# text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
# text_x = 200 + (230 - text_size[0]) // 2  # Center the text horizontally
# text_y = 400 + 250 + 30  # Position the text below the rectangle
# cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness)  
# border_color = (0, 255, 0)  # Green color (BGR)
# border_thickness = 10  # Thickness of the border
# image_with_border = cv2.copyMakeBorder(img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)

# cv2.imshow('image',image_with_border)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# q3
# import cv2
# img=cv2.imread('img1.jpg ')
# img2=cv2.imread('img2.jpg ')
# sub=cv2.subtract(img,img2)
# cv2.imshow('image',sub)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ##################sec2 
# # Import necessary libraries
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Step 1: Load the dataset
# iris = load_iris()
# X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
# y = iris.target  # Labels (species of iris)

# # Step 2: Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Step 3: Preprocess the data (standardize features)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
# X_test = scaler.transform(X_test)  # Transform the test data using the same scaler

# # Step 4: Train a machine learning model (Logistic Regression)
# model = LogisticRegression(max_iter=200)  # Increase max_iter for convergence
# model.fit(X_train, y_train)

# # Step 5: Make predictions on the test set
# y_pred = model.predict(X_test)

# # Step 6: Evaluate the model's performance
# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=iris.target_names))

# # Confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# ##########sec3 
from skimage import io, color, transform
# import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the image
image = io.imread('Cat_November_2010-1a.jpg')  # Replace with your image path

# Step 2: Display the original image
io.imshow(image)
io.show()

# Step 3: Crop the top half of the image
height = image.shape[0]
cropped_image = image[:height // 2, :]  # Crop the top half

gray_image = color.rgb2gray(cropped_image)

gray_image_scaled = (gray_image * 255).astype(np.uint8)
gray_image_scaled[gray_image_scaled > 200] = 255  

resized_image = transform.resize(gray_image_scaled, (gray_image_scaled.shape[0] // 2, gray_image_scaled.shape[1] // 2))

io.imshow(resized_image)
io.show()