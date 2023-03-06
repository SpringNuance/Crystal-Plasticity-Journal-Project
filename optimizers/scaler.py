from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
import numpy as np
# X = np.array([[1.00e+00, 2.00e+02],
#             [1.01e+00, 1.00e+01],
#             [1.00e+02, 2.00e+03],
#             [1.00e+00, 1.00e+03],
#             [1.00e-03, 2.00e+00],
#             [1.00e-03, 2.00e+00],
#             [1.00e-02, 2.50e+01],
#             [1.00e-03, 2.00e+00],
#             [1.00e-03, 3.00e+00],
#             [1.00e-03, 3.00e+00]])

X = np.array([[1.00e+00, 1.01e+00, 1.00e+02, 1.00e+00, 1.00e-03, 1.00e-03, 1.00e-02, 1.00e-03, 1.00e-03, 1.00e-03],
              [2.00e+02, 1.00e+01, 2.00e+03, 1.00e+03, 2.00e+00, 2.00e+00, 2.50e+01, 2.00e+00, 3.00e+00, 3.00e+00]])

X_one_dimension = np.array([1.00e+00, 1.01e+00, 1.00e+02, 1.00e+00, 1.00e-03, 1.00e-03, 1.00e-02, 1.00e-03, 1.00e-03, 1.00e-03])

powerTest = np.array([1, 1.3, 1.3, 1.3, 1, 1.3, 1.3, 1.3, 1.3, 1.3])



# X_root = root_transform(X, powerTest)
# print(X_root)
# X_pow = power_transform(X_root, powerTest)
# print(X_pow)

# X_root = root_transform(X_one_dimension, powerTest)
# print(X_root)
# X_pow = power_transform(X_root, powerTest)
# print(X_pow)

def CustomScaler(X, powerList):

    def root_transform(X, powerList):
        if len(X.shape) == 1:
            # X is a 1D array
            return X ** (1/powerList)
        else:
            # X is a 2D array
            
            return np.power(X, (1/powerList).reshape(1, -1))

    def power_transform(X, powerList):
        if len(X.shape) == 1:
            # X is a 1D array
            return X ** powerList
        else:
            # X is a 2D array
            return np.power(X, powerList.reshape(1, -1))
    
    poly_transformer = FunctionTransformer(func=root_transform,inverse_func=power_transform, kw_args={'powerList': powerList}, inv_kw_args={'powerList': powerList})
    # define the pipeline with steps for log transformation, standardization, and min-max scaling
    scaler = Pipeline([
        ('poly_transformer', poly_transformer),
        ('standardize', StandardScaler()),
        #('min_max_scale', MinMaxScaler(feature_range=rangeTuple)),
    ])
    scaler.fit(X)
    
    return scaler

    
# pipeline.fit(X)

# print(X)
# # apply the pipeline to a feature matrix X
# X_transformed = pipeline.transform(X)

# print(X_transformed)

# # apply the reverse pipeline to the transformed feature matrix X_transformed
# X_original = pipeline.inverse_transform(X_transformed)

# print(X_original)