import numpy as np
from sklearn.decomposition import PCA

class ITQ:
    def __init__(self, n_bits=64, n_iter=50):
        """
        Iterative Quantization (ITQ) 哈希算法
        
        参数:
            n_bits: 哈希码位数
            n_iter: 迭代次数
        """
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.pca = PCA(n_components=n_bits)
        self.R = None  # 旋转矩阵

    def fit(self, X):
        """
        训练ITQ模型
        
        参数:
            X: 输入数据矩阵 (n_samples, n_features)
        """
        # 数据预处理
        X = X - np.mean(X, axis=0)
        
        # PCA降维
        V = self.pca.fit_transform(X)
        
        # 初始化随机旋转矩阵
        np.random.seed(123)
        R = np.random.randn(self.n_bits, self.n_bits)
        U, _, _ = np.linalg.svd(R)
        R = U[:, :self.n_bits]
        
        # 迭代优化
        for _ in range(self.n_iter):
            Z = np.dot(V, R)
            B = np.sign(Z)
            C = np.dot(B.T, V)
            UB, sigma, UA = np.linalg.svd(C)
            R = np.dot(UA, UB.T)
        
        self.R = R
    
    def transform(self, X):
        """
        生成二进制哈希码
        
        参数:
            X: 输入数据 (n_samples, n_features)
            
        返回:
            二进制哈希码 (n_samples, n_bits)
        """
        if self.R is None:
            raise ValueError("Model not trained yet. Call fit() first.")
            
        V = self.pca.transform(X - np.mean(X, axis=0))
        Y = np.dot(V, self.R)
        return (Y > 0).astype(int)