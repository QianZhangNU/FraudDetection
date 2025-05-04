import pandas as pd
from scipy.stats import pearsonr, zscore

class shift_difference_transformer():
    def __init__(self, ref_id_names):
        self.ref_id_names = ref_id_names

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        for _, ref_id in enumerate(self.ref_id_names):
            X = self.shift_difference(X, ref_id)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def shift_difference(self, X, ref_id):
        fea_name = ref_id + '_SD'
        X[fea_name] = X[ref_id].diff().fillna(0)
        X[fea_name] = abs(X[fea_name])
        return X


class generate_zscore_transformer():
    def __init__(self, ref_id_names):
        self.ref_id_names = ref_id_names

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        for _, ref in enumerate(self.ref_id_names):
            X = self.generate_zscore(X, ref[0], ref[1])
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def generate_zscore(self, X, ref_id, comb_id):
        fea_name = ref_id + '_' + comb_id + '_zscore'
        X[fea_name] = X.groupby(comb_id)[ref_id].transform(lambda x: round(zscore(x, ddof=1)))
        X[fea_name].fillna(0, inplace=True)
        return X


class combine_category_transformer():
    def __init__(self, Lsts_name):
        self.Lsts_name = Lsts_name

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        for Lsts in self.Lsts_name:
            X = self.combine_category(X, Lsts)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def combine_category(self, X, Lsts):
        fea_name = ''
        for lst in Lsts:
            fea_name += lst + '_'
        X[fea_name] = ''
        for ii in range(0, len(Lsts), 2):
            if ii != len(Lsts) - 1:
                X[fea_name] = X[fea_name] + X[Lsts[ii]].astype(str) + '_' + X[Lsts[ii+1]].astype(str) + '_'
            else:
                X[fea_name] = X[fea_name] + X[Lsts[ii]].astype(str)
        return X


class generate_difference_transformer():
    def __init__(self, ids):
        self.ids = ids

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        for ids_id in self.ids:
            X = self.generate_difference(X, ids_id[0], ids_id[1])
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def generate_difference(self, X, id_1, id_2):
        fea_name = id_1 + '_' + id_2 + '_diff'
        X[fea_name] = round((X[id_1] - X[id_2]) / (X[id_2] + 0.000001) * 100)
        X.loc[(X[fea_name] > 100), fea_name] = 100
        X.loc[(X[fea_name] < -100), fea_name] = -100
        return X