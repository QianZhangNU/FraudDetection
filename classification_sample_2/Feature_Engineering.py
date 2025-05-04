import numpy as np

class Ordinal_Transformer_lob():
    def fit(self, X, y=None):
        self.res = {"CCB": 1, np.nan: 0}

    def transform(self, X, y=None):
        list_a=[0]*len(X)
        for i in range(0,len(X)):
            list_a[i] = self.res[X[i]]
        return pd.DataFrame(list_a,columns=['Sender_lob_conv'])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    
class Frequency_Transformer():
    
    def __init__(self, name_f, name_Label):
        self.name_f = name_f
        self.Label = name_Label
        self.nf_name = name_f+"_ave"
        
    def fit(self, X, y=None):
        self.group = X.groupby(self.name_f)[self.Label].mean()
        self.med = np.nanmean(self.group.values)
 
        
    def transform(self, X, y=None):
        X[self.nf_name] = X[self.name_f].map(self.group)
        if X[self.nf_name].isna().sum()==0:
            return X
        else:
            if X[self.name_f].dtype=='float64':
                return self.numeric_fill(X, y)
            else:
                X[self.nf_name].fillna(self.med, inplace=True)
                return X
  #      X[self.nf_name]=np.log((X[self.nf_name]-0.00001)/(1-X[self.nf_name]+0.00001))
        
               
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def numeric_fill(self, X, y=None):
        nan_indices = X[self.nf_name].isna()
        values_in_feature2 = X.loc[nan_indices, self.name_f]
        list_a=[]
        num = 10
        for x in values_in_feature2.values:
            list_a.append(self.find_near(x,num))
        
        X[self.nf_name][nan_indices]=list_a
        return X
        
    def find_near(self, x, num):
        A_group = np.array(self.group.index)
        diff = np.abs(A_group-x)
        indices = np.argpartition(diff,num)[:num]
        nearest_values=A_group[indices]
        x = self.group[nearest_values].sum()/num
        return x
    
class Dummy_Transformer(object):

    def fit(self, X, y=None):
        self.keys = set(X)

    def transform(self, X, y=None):
        res = {}
        for key in self.keys:
            res[key] = [0]*len(X)
        for i, item in enumerate(X):
            if item in self.keys:
                res[item][i] = 1
        return pd.DataFrame(res)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)