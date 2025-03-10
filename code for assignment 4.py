import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score


# Loading the data
data_trail1 = pd.read_csv('D:/Lectures/Industrial AI/Assignment 4/Trail1_extracted_features_acceleration_m1ai1-1.csv')
data_trail2 = pd.read_csv('D:/Lectures/Industrial AI/Assignment 4/Trail2_extracted_features_acceleration_m1ai1.csv')
data_trail3 = pd.read_csv('D:/Lectures/Industrial AI/Assignment 4/Trail3_extracted_features_acceleration_m2ai0.csv')

# Combining the datasets into one dataset
combined_dataset = pd.concat([data_trail1, data_trail2, data_trail3], ignore_index=True)

# Removing the columns
remove_columns = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']
combined_dataset.drop(columns=remove_columns, inplace=True)

# Replacing 'normal' events with 0 and other events with 1
combined_dataset['event'] = combined_dataset['event'].apply(lambda x: 0 if x == 'normal' else 1)

# For data normalization excluding the 'event' column
exclude_event = combined_dataset.drop(columns=['event'])

scaler = MinMaxScaler()

normalized_datasets = scaler.fit_transform(exclude_event)

# Adding event column into the normalized dataset
normalized_df = pd.DataFrame(normalized_datasets, columns=exclude_event.columns)
normalized_df['event'] = combined_dataset['event']



normalized_df.to_csv('C:/Users/hp/Documents/normalized_data.csv', index=False)

normalized_data = pd.read_csv('C:/Users/hp/Documents/normalized_data.csv')


# splitting dataset into 80/20 ratio for training and testing
a = normalized_df.drop(columns=['event'])
b = normalized_df['event']
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=10)


print(f"Training : {a_train.shape}, Testing: {a_test.shape}")
print(f"Training Labels: {b_train.shape}, Test Labels: {b_test.shape}")


model = LogisticRegression()

# Training the model with training data (a_train, b_train)
model.fit(a_train, b_train)

# Predicting the events 
predictions = model.predict(a_test)


print(predictions[:10])

accuracy = accuracy_score(b_test, predictions)
precision = precision_score(b_test, predictions)


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")


