import matplotlib
matplotlib.use('Qt5Agg')  # Use o backend Qt5Agg para plotagem interativa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class AdalineGD(object):
    """
    Parameters
    --------------
    eta: float
        Taxa de aprendizado (entre 0.0 e 1.0)
    n_iter: int
        Passagens sobre o conjunto de dados de treinamento
    random_state: int
        Semente do gerador de números aleatórios para inicialização aleatória dos pesos
    
    Attributes
    --------------
    w_: 1d-array
        Pesos após o ajuste
    cost_: list
        Valor da função de custo de soma dos quadrados em cada época.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """ 
        Ajustar os dados de treinamento 
        

        X : {array-like}, shape = [n_samples, n_features]
            Vetores de treinamento, onde n_samples = número de amostras e n_features = número de características
            
        y : array-like, shape = [n_samples]
            Valores alvo
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[0] += self.eta * errors.sum()
            self.w_[1:] += self.eta * X.T.dot(errors)
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        
        return self
    
    def net_input(self, X):
        """Calcular a entrada líquida, retorna um array de valores que representa a entrada líquida para cada amostra."""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Calcular a ativação linear, retorno o mesmo array de entrada"""
        return X
    
    def predict(self, X):
        """Retornar o rótulo da classe após o passo unitário, retorna um array de rótulos de classe (1 ou -1) para cada amostra"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# Carregar a tabela de base de dados a partir do arquivo .txt
df = pd.read_csv('basedeobservacoes_trabalho06.txt', delim_whitespace=True)
print(df.head())  

y = df['y'].values
X = df[['x']].values

# Calcular a linha de regressão linear
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Adiciona x0 = 1 a cada instância
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Comparar os resultados da regressão com os obtidos utilizando as equações a e b
y_pred = X_b.dot(theta_best)
a = theta_best[1]
b = theta_best[0]

# Calcular o coeficiente de correlação de Pearson
corr, _ = pearsonr(y, y_pred)

# Calcular o coeficiente de determinação (R^2)
ss_total = ((y - y.mean()) ** 2).sum()
ss_residual = ((y - y_pred) ** 2).sum()
r_squared = 1 - (ss_residual / ss_total)

print(f"Coeficiente de correlação de Pearson: {corr}")
print(f"Coeficiente de determinação (R^2): {r_squared}")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Treinar o modelo Adaline com taxa de aprendizado 0.01
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log(ada1.cost_), marker='o')
ax[0].set_xlabel('Épocas')
ax[0].set_ylabel('log(Soma dos erros quadrados)')
ax[0].set_title('Adaline - Taxa de aprendizado 0.01')

# Treinar o modelo Adaline com taxa de aprendizado 0.0001
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1), np.log(ada2.cost_), marker='o')
ax[1].set_xlabel('Épocas')
ax[1].set_ylabel('Soma dos erros quadrados')
ax[1].set_title('Adaline - Taxa de aprendizado 0.0001')

plt.show()