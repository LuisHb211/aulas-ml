import matplotlib
matplotlib.use('Qt5Agg')  # Use o backend Qt5Agg para plotagem interativa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdalineGD(object):
    """ Classificador ADAptive LInear NEuron
    
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
        """ Ajustar os dados de treinamento 
        
        Parameters
        -------------
        X : {array-like}, shape = [n_samples, n_features]
            Vetores de treinamento, onde n_samples = número de amostras e n_features = número de características
            
        y : array-like, shape = [n_samples]
            Valores alvo
            
        Returns
        -----------
        self: object
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
        """Calcular a entrada líquida"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Calcular a ativação linear"""
        return X
    
    def predict(self, X):
        """Retornar o rótulo da classe após o passo unitário"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# Carregar o arquivo Excel
df = pd.read_excel('C:/REPOS/Pessoais/aulas-ml/Basedados_B2.xlsx')
print(df.head())  # Verificar se o DataFrame foi carregado corretamente

# Extrair rótulos e características
y = df['t'].values
X = df[['s1', 's2']].values

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