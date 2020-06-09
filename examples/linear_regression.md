Examples of using my_machine_learning.models.linear_regression.LinearRegression.

# 1. Importing
```python
from my_machine_learning.models.linear_regression import LinearRegression

# define or load data here
          ...
          ...
          
lr = LinearRegression()
lr.fit(X_train, y_train)
```

# 2. Evaluating
```python
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
```

# 3. Using hyperparameters
```python
lr.fit(X_train, y_train, epochs=2000, learning_rate=10, verbose=True)
```

# 4. History of fitting
```python
history = lr.fit(X_train, y_train, epochs=200)
```

# 5. Save weights to file
```python
lr.to_file('model.bin')
```

# 6. Save weights to np.ndarray
```python
weights = lr.weights()
```

# 7. Load weights from file
```python
lr = LinearRegression.from_file('model.bin)
```

# 8. Load weights from array
```python

arr = [0.125, 0.1, ..., 0.25]
lr = LinearRegression.from_array(arr)
```
