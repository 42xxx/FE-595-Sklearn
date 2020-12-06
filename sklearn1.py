from sklearn import datasets
from sklearn import linear_model
import pandas as pd
import numpy as np

# part 1
bos = datasets.load_boston()
boston = pd.DataFrame(bos.data)
boston.columns = bos.feature_names
boston['price'] = bos.target

lin_reg = linear_model.LinearRegression()
y = boston['price']
x = boston[bos.feature_names]
a = lin_reg.fit(x, y)
res_df = pd.DataFrame()
res_df['beta'] = np.abs(a.coef_)
res_df['variable'] = bos.feature_names
res_df = res_df.sort_values(by='beta', ascending=False)
print(res_df)

# part 2