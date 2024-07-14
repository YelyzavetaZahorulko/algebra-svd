import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

#  Читаємо дані
ratings_file_path = 'ratings.csv'
movies_file_path = 'movies.csv'

ratings_df = pd.read_csv(ratings_file_path)
movies_df = pd.read_csv(movies_file_path)

# Робимо матрицю
ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')

# Видаляємо певні данні(1- користувачі, що оцінили менше 10 фільмів, 2 - фільми, що мають менше 20 оцінок
ratings_matrix = ratings_matrix.dropna(thresh=10, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=1)

# Обробка даних для SVD
ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

print("Shape of the NumPy array R:", R.shape)

# Різні числа k
k_values = [3]
results = []

for k in k_values:
    U, sigma, Vt = svds(R_demeaned, k=k)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

    # Calculate RMSE
    original_ratings = ratings_matrix.values
    predicted_ratings = preds_df.values
    mask = ~np.isnan(original_ratings)
    rmse = np.sqrt(np.mean((predicted_ratings[mask] - original_ratings[mask]) ** 2))
    results.append((k, rmse))

    print(f"k={k}, RMSE={rmse}")

k = 20
U, sigma, Vt = svds(R_demeaned, k=k)
sigma = np.diag(sigma)


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

original_ratings = ratings_matrix.values
predicted_ratings = preds_df.values
predicted_only_ratings = predicted_ratings.copy()
predicted_only_ratings[~np.isnan(original_ratings)] = np.nan
predicted_only_ratings_df = pd.DataFrame(predicted_only_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)


# Функція, що рекомендує фільми


def recommend_movies(user_id, num_recommendations=10):
    user_row_number = user_id - 1  # Припускаючи, що user_id починається з 1
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)

    # Отримання даних користувача та об'єднання з даними про фільми
    user_data = ratings_df[ratings_df.userId == user_id]
    user_full = (user_data.merge(movies_df, how='left', left_on='movieId', right_on='movieId').
                 sort_values(['rating'], ascending=False))

    # Рекомендація фільмів з найвищою прогнозованою оцінкою, які користувач ще не бачив
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='movieId',
                             right_on='movieId').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('movieId', ascending=False).
                       iloc[:num_recommendations, :-1])

    return recommendations

userId = 2
recommended_movies = recommend_movies(userId, 10)

recommended_movies_path = '/Users/yelyzaveta/PycharmProjects/algebra-svd/recommended_movies.csv'
recommended_movies.to_csv(recommended_movies_path, index=False)
print(recommended_movies)
