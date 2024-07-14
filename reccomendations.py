import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

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


# Замінюємо NaN данні на середнє значення (2.5)
ratings_matrix_filled = ratings_matrix.fillna(2.5)

# Перетворимо PandasDF на масив NumPy
R = ratings_matrix_filled.values

# Середнє значення для кожного користувача
user_ratings_mean = np.mean(R, axis=1)

# Прибираємо судження
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
print("Розмір NumPY масиву:", R.shape)

# SVD
U, sigma, Vt = svds(R_demeaned, k=3)
print("U matrix:\n", U)

# Візуалізація оцінки користувачів
U_20 = U[:20]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U_20[:, 0], U_20[:, 1], U_20[:, 2], c='r', marker='o')

ax.set_title('User Feature Space')

plt.show()
