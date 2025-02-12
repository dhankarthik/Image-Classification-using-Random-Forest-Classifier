# Declaring the Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset

df = pd.read_csv("file1.csv")

# Display the first image

a = df.iloc[3, 1:].values
a = np.pad(a, (0, 1), 'constant')  # Add a dummy value
a = a.reshape(28, 28).astype('uint8')
plt.imshow(a)
plt.show()

# Split the dataset into features and labels

df_x = df.iloc[:, 1:]
df_y = df.iloc[:, 0]

# Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

# Initialize and train a random forest classifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

# Make predictions on the test set

pred = rf.predict(x_test)

# Evaluate the accuracy of the model

correct_count = np.sum(pred == y_test.values)
accuracy = correct_count / len(pred)

# Print the results

print("Predictions:", pred)
print("Correct Count:", correct_count)
print("Total Samples:", len(pred))
print(f"Accuracy: {accuracy * 100:.2f}%")
