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
merged.head()
# %%
merged.info()
#%%
merged.drop(columns = 'timestamp', inplace=True)
# %%
def extract_primary_genre(genres):
    genre_list = genres.split('|')
    return genre_list[0] if len(genre_list) > 0 else None

merged['PrimaryGenre'] = merged['genres'].apply(extract_primary_genre)

merged.drop(columns=['genres'], inplace=True)

merged.head()
#%%
#mean ratings
# Calculate the weighted average rating for each movie based on its primary genre
movie_avg_ratings = merged.groupby(['movieId', 'title', 'PrimaryGenre'])['rating'].agg(['mean', 'count']).reset_index()
movie_avg_ratings.columns = ['movieId', 'title', 'PrimaryGenre', 'mean_rating', 'rating_count']
# Merge the calculated ratings back into the main dataset
merged = pd.merge(merged, movie_avg_ratings, on=['movieId', 'title', 'PrimaryGenre'])

#%%
# Display the updated dataset
merged.head()
#%%
# Drop duplicate entries
merged.drop_duplicates(subset=['movieId', 'title', 'PrimaryGenre'], inplace=True)
#%%
merged
#%%
#filtering movies that are not watched much
merged = merged[merged['rating_count'] >= 75]
#%%
merged
#%%
#again filtering out those movies which do not have such watch rate
merged = merged[merged['mean_rating'] >= 2.5]
#%%
merged
#%%
merged['movieId_numeric'] = pd.to_numeric(merged['movieId'], errors='coerce')
#%%
nan_values = merged['movieId_numeric'].isna().sum()
#%%
nan_values

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from sklearn.preprocessing import LabelEncoder
#%%
features = merged[['movieId', 'title', 'PrimaryGenre']]
target = merged['mean_rating']
#%%
# Drop rows with missing values
features = features.dropna(subset=['movieId', 'title', 'PrimaryGenre'])
#%%

features['movieId'] = pd.to_numeric(features['movieId'], errors='coerce')
#%%
# Drop rows with NaN in 'movieId'
features = features.dropna(subset=['movieId'])

# One-hot encode 'PrimaryGenre'
genre_dummies = pd.get_dummies(features['PrimaryGenre'], prefix='genre', prefix_sep='_')
features = pd.concat([features, genre_dummies], axis=1)
#%%
# Drop unnecessary columns
features.drop(['movieId', 'PrimaryGenre'], axis=1, inplace=True)
#%%
# Convert 'title' to integer labels using LabelEncoder
label_encoder = LabelEncoder()
features['title'] = label_encoder.fit_transform(features['title'])
#%%
# Define embedding size for 'title'
embedding_size_title = 10
#%%
# Create an embedding layer for 'title'
input_title = Input(shape=(1,))
embedding_layer_title = Embedding(input_dim=len(features['title'].unique()) + 1, output_dim=embedding_size_title, input_length=1)(input_title)
flatten_layer_title = Flatten()(embedding_layer_title)
#%%
# Define input layers for the neural network
input_genre = Input(shape=(len(genre_dummies.columns),))
#%%
# Concatenate all input layers
concatenated_inputs = Concatenate()([input_genre, flatten_layer_title])
#%%
# Define the neural network model
dense_layer_1 = Dense(32, activation='relu')(concatenated_inputs)
output_layer = Dense(1, activation='linear')(dense_layer_1)
#%%
# Create the model
model = Model(inputs=[input_genre, input_title], outputs=output_layer)
#%%
# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')
#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#%%
# Train the model
model.fit([X_train.iloc[:, 1:], X_train['title']], y_train, epochs=80, batch_size=32, validation_split=0.2)
#%%
# Evaluate the model on the test set
predictions = model.predict([X_test.iloc[:, 1:], X_test['title']])
mse = mean_squared_error(y_test, predictions)

#%%
mse

