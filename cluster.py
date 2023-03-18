import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Load the CSV data
csv_file = "test.csv"
data = pd.read_csv(csv_file)

# Assuming the text column is named 'text'
text_data = data['null']

# # Preprocess the textual data (optional)
# # This step can include removing stopwords, stemming, or other text preprocessing techniques
# # Customize this function as needed
# def preprocess_text(text):
#     # Add your preprocessing steps here
#     return text

# preprocessed_text_data = text_data.apply(preprocess_text)

# Transform the textual data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_data)

# Normalize the data
normalized_tfidf_matrix = normalize(tfidf_matrix)

# Perform k-means clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, max_iter=100, random_state=42)
kmeans.fit(normalized_tfidf_matrix)

# Assign the cluster labels to the original data
data['cluster_label'] = kmeans.labels_

# Sort by cluster label
data.sort_values(by='cluster_label', inplace=True)

# Write data to a new CSV file
output_csv_file = csv_file.split(".csv")[0] + "_clustered.csv"
data.to_csv(output_csv_file, index=False)