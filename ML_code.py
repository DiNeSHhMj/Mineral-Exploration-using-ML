
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import cv2 
import numpy as np

# Load the satellite images
gravity = cv2.imread(r'Gravity Clipped.tif', cv2.IMREAD_GRAYSCALE)
magnetics = cv2.imread(r'Magnetics Clipped.tif', cv2.IMREAD_GRAYSCALE)
radiometrics = cv2.imread(r'Radiometrics.tif', cv2.IMREAD_GRAYSCALE)
geology = cv2.imread(r'Geology Raster.tif', cv2.IMREAD_GRAYSCALE)
areas_of_interest = cv2.imread(r'Areas of Interest.tif', cv2.IMREAD_GRAYSCALE)
distance_to_fault = cv2.imread(r'Distance to Fault.tif', cv2.IMREAD_GRAYSCALE)

# Check if the images are loaded properly
if gravity is None:
    print('Error: Could not read gravity image')
if magnetics is None:
    print('Error: Could not read magnetics image')
if radiometrics is None:
    print('Error: Could not read radiometrics image')
if geology is None:
    print('Error: Could not read geology image')
if distance_to_fault is None:
    print('Error: Could not read distance_to_fault image')
if areas_of_interest is None:
    print('Error: Could not read areas_of_interest image')

#Analysing the shape of each numpy array,
print('Radiometrics shape is',radiometrics.shape)
print('Magnetics shape is',magnetics.shape)
print('Gravity shape is',gravity.shape)
print('Geology shape is',geology.shape)
print('Distance to Fault shape is',distance_to_fault.shape)
print('Area of Interest shape is',areas_of_interest.shape)

#Analysing radiometrics array
print(radiometrics)

# # Concatenate all the images together
images = np.stack((gravity, magnetics, radiometrics, geology, distance_to_fault , areas_of_interest), axis=2)


# Convert the images to a 1D array of features
features = images.reshape(-1, images.shape[-1])

# Create a pandas DataFrame with the features and add column names
df = pd.DataFrame(features)
df.columns = ['gravity', 'magnetics', 'radiometrics', 'geology', 'distance_to_fault', ' areas_of_interest']

# Define a function to engineer new features
def add_features(df):
    df['gravity_magnetics'] = df['gravity'] * df['magnetics']
    df['radiometrics_geology'] = df['radiometrics'] * df['geology']
   # df['distance_to_fault_areas_of_interest'] = df['distance_to_fault'] * df['areas_of_interest']
    return df
# Apply the feature engineering function
df = add_features(df)

# Define the target variable
target = areas_of_interest.reshape(-1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predict on the test data and evaluate the model performance
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Use the trained model to predict mineral deposits in the study area
predicted_deposits = rfc.predict(df)

# Reshape the predicted deposits back into an image
predicted_deposits = predicted_deposits.reshape(gravity.shape)

#
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Predicted Mineral Deposits')

# create a ScalarMappable object with reversed normalization
cmap = plt.cm.gist_rainbow_r 
norm = plt.Normalize(vmin=geology.max(), vmax=geology.min()) 
scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

ax.imshow(geology, cmap=cmap, norm=norm)
y, x = np.where(predicted_deposits == 1)
for yy, xx in zip(y, x):
    circ = plt.Circle((xx, yy), radius=10, color='r', fill=False, linewidth=2)
    ax.add_patch(circ)


cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  
cbar = fig.colorbar(scalar_map, cax=cax)  
cbar.set_label('Area Of Interest')


plt.show()

