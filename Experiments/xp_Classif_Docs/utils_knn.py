import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


def get_acc_knn(d_sw, y, idx_train, idx_test):
    L_acc = []
    L_f1_macro = []
    L_f1_micro = []

    for k in range(len(idx_train)):   
        y_train = y[idx_train[k]]
        y_test = y[idx_test[k]]

        d_train_sw = d_sw[np.ix_(idx_train[k], idx_train[k])]
        d_test_sw = d_sw[np.ix_(idx_test[k], idx_train[k])]

        knn = KNeighborsClassifier(metric="precomputed")

        param_grid = {"n_neighbors": np.arange(1, 25)}
        knn_gscv = GridSearchCV(knn, param_grid, cv=5)
        knn_gscv.fit(d_train_sw, y_train)
        sw_predict = knn_gscv.predict(d_test_sw)

        acc = accuracy_score(y_test, sw_predict)
        f1_macro = f1_score(y_test, sw_predict, average='macro')
        f1_micro = f1_score(y_test, sw_predict, average='micro')

        L_acc.append(acc)
        L_f1_macro.append(f1_macro)
        L_f1_micro.append(f1_micro)

    return L_acc


class Features(BaseEstimator, TransformerMixin):
    
    def __init__(self, rho=1, k=0, loss="rsw", dataset="BBC"):
        self.rho = rho
        self.k = k
        self.loss = loss
        self.dataset = dataset
    
    def fit(self, idx, y=None):
        self.path = "./results_"+str(self.dataset)+"/"

        if self.loss == "rsw":
            d_sw = np.loadtxt(self.path + "d_projs500_rsw_"+self.dataset+"_rho1"+str(self.rho)+"_rho2"+str(self.rho)+"_k"+str(self.k))        
        elif self.loss == "suw":
            d_sw = np.loadtxt(self.path + "d_projs500_suw_"+self.dataset+"_rho1"+str(self.rho)+"_rho2"+str(self.rho)+"_k"+str(self.k))
        elif self.loss == "sinkhorn":
            d_sw = np.loadtxt(self.path + "d_sinkhorn_"+self.dataset+"_rho"+str(self.rho)+"_reg0.1_k0")

            
        self.d_sw = d_sw
        self.idx_train = idx
        
        return self
    
    def transform(self, idx, y=None): 
        X = self.d_sw[np.ix_(idx, self.idx_train)]
        return X
    
    def get_params(self, deep=True):
        return {"rho": self.rho, "k": self.k, "loss": self.loss, "dataset": self.dataset}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
def get_acc_knn_cv_rho(dataset, method, k, rhos, y, idx_train, idx_test):
    L_acc = []    

    for i in range(len(idx_train)):
    
        y_train = y[idx_train[i]]
        y_test = y[idx_test[i]]

        knn_gscv = GridSearchCV(
            make_pipeline(
                Features(),
                GridSearchCV(
                    KNeighborsClassifier(metric="precomputed"),
                    param_grid = {"n_neighbors": np.arange(1, 25)}
                )
            ),
            {"features__rho": rhos, "features__k": [k], "features__loss": [method], "features__dataset": [dataset]}
        )

        knn_gscv.fit(idx_train[i], y_train)
        sw_predict = knn_gscv.predict(idx_test[i])
        
        acc = accuracy_score(y_test, sw_predict)
        L_acc.append(acc)
        
    return L_acc