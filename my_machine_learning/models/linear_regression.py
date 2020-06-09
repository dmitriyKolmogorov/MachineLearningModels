import numpy as np
from sklearn.metrics import r2_score


# TODO: add regularization and sample weights
# TODO: realize self-made version of r2_score
# TODO: create normalization and intercept
class LinearRegression(object):
    '''
    Regression model without multiprocessing capability for regression.
    Uses least-squares method for fitting model.

    WARNING! Normalize data before to use this model, because you can get RuntimeWarning and model will not work correctly. 

    Appeared in version 0.0.1 .
    '''

    def __call__(self, X) -> np.ndarray:
        '''
        Another program interface for LinearRegression.predict()
        '''
        return self.predict(X)


    def fit(self, X, Y, epochs=20, learning_rate=1, verbose=False) -> list:
        '''
        Method for fitting model on training data.

        Parameters:
            X: array-like object with shape (n_samples, n_features);
            y: array-like object with shape (n_samples, );
            epochs:int, default=None: num of epochs for fitting; 
            learning_rate:float, default=1: learning rate for gradient descent. Learning rate is in range (0, 1);
            verbose:bool, default=False: if verbose is True, report will be printed for every epochs. You can use any Python object with __bool__ method as value of this argument;

        Returns:
            history:list, list with losses for every epoch. It has shape like (epochs, );
        '''

        # input data validation
        try:
            X = np.array(X, dtype=np.longdouble)
        except Exception as ex:
            raise TypeError(f'Can not convert X to np.ndarray ({ex}).')

        try:
            Y = np.array(Y, dtype=np.longdouble)
        except Exception as ex:
            raise TypeError(f'Can not convert X to np.ndarray ({ex}).')

        # X has not shape like (n_samples, n_features)
        if len(X.shape) != 2:
            raise ValueError(f'X has wrong shape - {X.shape}. Expected shape is like (n_samples, n_features).')

        # y has not shape like (n_samples, )
        if len(Y.shape) != 1:
            raise ValueError(f'Y has wrong shape - {X.shape}. Expected shape is like (n_samples, ).')

        if Y.shape[0] != X.shape[0]:
            raise ValueError(f'X and Y has different number of items. {X.shape} vs {Y.shape}')

        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError('Argument epochs is not integer or less than zero.')

        # model's weights initializating
        self.__w = np.random.random(X.shape[1]).astype(np.longdouble)

        history = list()

        # model's fitting
        for epoch in range(epochs):

            # get residual
            residual = self.__calculate_residual(X, Y)

            # weights update
            self.__w -= learning_rate * np.dot(X.T, residual) / len(X)

            # calculate loss
            loss = (residual ** 2).sum() / (2 * len(X))
            history.append(loss)

            # print report
            if verbose:
                print(f'Epoch {epoch + 1}. Loss: {loss}')

        return history

            
    def __calculate_residual(self, X, y) -> np.ndarray:
        '''
        "Private" method for calculating difference between predicted values and true values.
        '''
        predictions = self.predict(X)
        
        return predictions - y


    def __normalize(self, X) -> np.ndarray:
        '''
        "Private" method for normalization.
        '''

        X_norm = np.linalg.norm(X)

        return X / X_norm

    
    def __set_weights(self, X):
        '''
        "Private" method created to set model's weights.

        Returns self isinstance.
        '''

        self.__w = np.array(X)

        return self


    def predict(self, X) -> np.ndarray:
        '''
        Method for model's prediction. 

        Parameters:
            X: array-like object with shape (n_samples, n_features);

        Returns:
            X_predicted: np.ndarray object with shape (n_samples, );
        '''

        X = np.array(X)

        return X.dot(self.__w) 

    
    def weights(self) -> np.ndarray:
        '''
        Method created to get model's weights.

        Returns:
            w:np.ndarray. Weights of model.
        '''

        try:
            return self.__w
        except AttributeError:
            raise AttributeError('The model was not fitted. Use LinearRegression.fit() to fix it.')


    def score(self, X, y) -> float:
        '''
        Method for avaluating regression model.

        Parameters:
            X: array-like object with shape (n_samples, n_features);

        Returns:
            scores: float. R2 score between y and predicted by model values;
        '''

        return r2_score(self.predict(X), y)


    def to_file(self, filename:str) -> None:
        '''
        Method created to save model's weights to text or binary file.

        Parameters:
            filename:str, name of the file;

        Returns:
            None;
        '''

        self.__w.tofile(filename)

    
    @classmethod
    def from_file(cls, filename:str):
        '''
        LinearRegression.from_file() loads weights for model from binary or text file.

        The best way is to save model's weights with LinearRegression.to_file() and then load it with LinearRegression.from_file.
        Example:

            >>>lr.fit(X, y)
            >>>lr.to_file('weights.bin')
            
            >>>lr = LinearRegression.from_file('weights.bin')

        Parameters:
            filename:str, name of the file;
        '''

        # load weights from file
        X = np.fromfile(filename)

        # validate input
        try:
            X = np.array(X, dtype=np.longdouble)
        except Exception as ex:
            raise TypeError(f'Can not convert X to np.ndarray ({ex}).')

        # X has not shape like (n_samples, n_features)
        if len(X.shape) != 2:
            raise ValueError(f'X has wrong shape - {X.shape}. Expected shape is like (n_samples, n_features).')

        return cls().__set_weights(X)


    @classmethod
    def from_array(cls, X):
        '''
        LinearRegression.from_array() loads weights for model from array.

        The best way is to save weights using LinearRegression.weights() and then load it with LinearRegression.from_array.
        Example:

            >>>lr.fit(X, y)
            >>>weights = lr.weights()

            >>>lr = LinearRegression.from_array(weughts)

        Parameters:
            X: array-like object with shape like (n_samples, n_features)
        '''

        # validate input
        try:
            X = np.array(X, dtype=np.longdouble)
        except Exception as ex:
            raise TypeError(f'Can not convert X to np.ndarray ({ex}).')

        # X has not shape like (n_samples, n_features)
        if len(X.shape) != 2:
            raise ValueError(f'X has wrong shape - {X.shape}. Expected shape is like (n_samples, n_features).')

        return cls().__set_weights(X)
