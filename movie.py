#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn as sk
import keras as ker
# %%
rating = pd.read_csv("/Users/swarnim/Desktop/ratings.csv")
genre = pd.read_csv("/Users/swarnim/Desktop/movies.csv")

# %%
# Read the movies_metadata CSV file, including only the relevant columns
merged = pd.merge(genre, rating, on='movieId')
# %%
merged
# %%
merged.info()
# %%
merged.hist(figsize=(15,8))
# %%
def extract_primary_genre(genres):
    genre_list = genres.split('|')
    return genre_list[0] if len(genre_list) > 0 else None

# Apply the function to create a new 'PrimaryGenre' column
merged['PrimaryGenre'] = merged['genres'].apply(extract_primary_genre)

# Drop the original 'genres' column if no longer needed
merged.drop(columns=['genres'], inplace=True)

# Print the DataFrame with the primary genre
merged.head()
# %%
# Perform one-hot encoding for genres
genre_dummies = pd.get_dummies(merged['PrimaryGenre'], prefix='genre', prefix_sep='_')
#%%
genre_dummies = genre_dummies.astype(int) #0s and 1s instead of stringa

#%%
# Concatenate the one-hot encoded genre columns with the original DataFrame
merged = pd.concat([merged, genre_dummies], axis=1)
# Drop the 'PrimaryGenre' column as it's no longer needed
merged.drop(columns=['PrimaryGenre'], inplace=True)

# %%
merged
# %%
x = merged.drop('title', axis=1)
y = merged['title']
#%%
# Calculate the total count of movies in each genre
genre_counts = merged.iloc[:, 5:].sum()  # Assuming the one-hot encoded genre columns start from the 5th column

# Create a bar graph
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar', color='steelblue')
plt.title('Number of Movies per Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()
# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# %%
x_train
# %%
y_train
#%%
#label encoding is done to convert string to int
# Import the LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit and transform the 'title' column in y_train
y_train_encoded = label_encoder.fit_transform(y_train.astype(str))
#%%
y_train_encoded

# %%
training = x_train.join(y_train)
# %%
training
# %%
training.hist(figsize=(15,8))
# %%
numeric_columns = training.select_dtypes(include=['number'])

# Create a correlation matrix for numeric columns
correlation_matrix = numeric_columns.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
# %%
training
# %%
training.drop(columns = 'timestamp', inplace = True)
x_test.drop(columns = 'timestamp', inplace = True)
# %%
training
# %%
from keras import Sequential
from keras.layers import Dense
neunet = Sequential()
# %%
neunet.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
neunet.add(Dense(32, activation='relu'))
neunet.add(Dense(1, activation='sigmoid'))

#turning y into 2D array
y_train_encoded = np.reshape(y_train_encoded, (-1, 1))
# %%
x_train = np.array(x_train)
y_train_encoded = np.array(y_train_encoded)
# %%
import tensorflow.keras as keras

from keras import optimizers
#%%
optimizer = keras.optimizers.Adam(learning_rate=0.001) 
neunet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# %%
y_train_encoded.dtype
# %%
final_network = neunet.fit(x_train, y_train_encoded, epochs=5, batch_size=64)
# %%
# Fit and transform the 'y_test' labels
y_test_encoded = label_encoder.fit_transform(y_test)
#%%

x_test = np.asarray(x_test)
y_test_encoded = np.asarray(y_test_encoded)
# %%
loss, accuracy = neunet.evaluate(x_test, y_test_encoded)
# %%
loss
# %%
accuracy
