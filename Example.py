import numpy as np
from sklearn.preprocessing import StandardScaler

# Original data
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)  # Reshape to (n_samples, n_features)

# Scale the data
scaler = StandardScaler()
scaled_Closed_records = scaler.fit_transform(data)

print("Scaled Data:\n", scaled_Closed_records)

##############################################################

# Initialize lists to hold sequences and labels
X_train = []
y_train = []

# Define the sequence length
sequence_length = 3

# Generate sequences
for i in range(sequence_length, len(scaled_Closed_records)):
    X_train.append(scaled_Closed_records[i - sequence_length:i, 0])
    y_train.append(scaled_Closed_records[i, 0])

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape X_train to (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_train:\n", X_train)
print("y_train:\n", y_train)

##############################################################

print("X_train:\n", X_train)
print("y_train:\n", y_train)
"""
X_train shape: (3, 3, 1)
y_train shape: (3,)

"""

print("Scaled Data:\n", scaled_Closed_records)

[[-1.41421356]
 [-0.70710678]
 [ 0.        ]
 [ 0.70710678]
 [ 1.41421356]]

##############################################################

##############################################################

print("X_train:\n", X_train)

"""
[[[−1.41421356],
  [−0.70710678],
  [ 0.        ]],

 [[−0.70710678],
  [ 0.        ],
  [ 0.70710678]],

 [[ 0.        ],
  [ 0.70710678],
  [ 1.41421356]]
  
"""

print("y_train:\n", y_train)

"""
[ 0.70710678, 1.41421356]

"""


##############################################################
