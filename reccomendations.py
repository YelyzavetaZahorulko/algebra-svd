import pandas as pd

# Зчитування CSV file
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# Вивід даних
print("Таблиця оцінювання фільмів:")
print(df.describe())

# Прибираємо користувачів, що оцінили мало фільмів
ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)

# Прибираємо фільми з малою к-стю оцінок
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)

print("Розміри матриці з корегованими даними:", ratings_matrix.shape)

