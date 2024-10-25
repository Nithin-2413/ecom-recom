# Import required libraries
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from collections import defaultdict

# Load dataset
# Assuming the data file contains columns: 'userId', 'productId', and 'rating'
file_path = '/content/drive/MyDrive/ecommerce_data.csv'  # Update the path as needed
data = pd.read_csv(file_path)

# Display first few rows
print("Data Preview:")
print(data.head())

# Preprocessing
# Keep only necessary columns and drop any missing values
data = data[['userId', 'productId', 'rating']].dropna()
data['userId'] = data['userId'].astype(int)
data['productId'] = data['productId'].astype(int)

# Load data into surprise library format
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['userId', 'productId', 'rating']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Initialize SVD (Singular Value Decomposition) model
model = SVD()

# Train model on training data
model.fit(trainset)

# Evaluate the model with cross-validation
cross_val_results = cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Function to get top-N recommendations for each user
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort predictions for each user and retrieve the N highest-rated items
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Predict ratings for the test set
predictions = model.test(testset)

# Get top-10 recommendations for each user
top_n_recommendations = get_top_n(predictions, n=10)

# Display recommendations for the first few users
print("\nTop-10 Product Recommendations:")
for uid, user_ratings in list(top_n_recommendations.items())[:5]:
    print(f"\nUser {uid}:")
    for iid, rating in user_ratings:
        print(f"  Product {iid} with estimated rating {rating:.2f}")

# Evaluation: Calculate RMSE and MAE for the model
rmse = cross_val_results['test_rmse'].mean()
mae = cross_val_results['test_mae'].mean()
print(f"\nAverage RMSE: {rmse:.2f}")
print(f"Average MAE: {mae:.2f}")

# Recommendations Function: Generate recommendations for a specific user
def recommend_for_user(user_id, n=5):
    # Get all products not rated by the user
    user_data = data[data['userId'] == user_id]
    all_products = data['productId'].unique()
    rated_products = user_data['productId'].unique()
    unrated_products = [p for p in all_products if p not in rated_products]

    # Predict ratings for each unrated product
    predictions = [model.predict(user_id, product) for product in unrated_products]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Return top N recommendations
    return [(pred.iid, pred.est) for pred in predictions[:n]]

# Example: Recommend products for a specific user
user_id = 1  # Replace with a valid user ID from your dataset
recommended_products = recommend_for_user(user_id, n=5)

print(f"\nTop recommendations for User {user_id}:")
for product_id, estimated_rating in recommended_products:
    print(f"  Product {product_id} with estimated rating {estimated_rating:.2f}")
