<p align="right">
  <img src="http://meusite.mackenzie.br/rogerio/mackenzie_logo/UPM.2_horizontal_vermelho.jpg" width="30%" align="center"/>
</p>

# Inteligência Artificial - Resumo N1

rogerio.oliveira@mackenzie.br  

<br>

<br>



**1** [**Introdução à Inteligência Artificial: conceito, história e paradigmas**](https://colab.research.google.com/github/Rogerio-mack/Inteligencia_Artificial/blob/main/IA_Introducao.ipynb) 

- IA $\times$ ML $\times$ Deep Learning
- IA Fraca $\times$ IA Forte
- ML $\times$ Data Science
- Aplicações ou Tarefas: **Regressão, Classificação,** Clusterização, Regras de Associação, Detecção de Anomalias, Matching etc.
- CRISP DM, 6 fases, **não linear**
- Aprendizado: 
  - **Supervisionado:** **Regressão, Classificação** (Conjunto de Treinamento: Exemplos)
  - Não Supervisionado
  - Com Reforço 

**2** [**Python básico para o Aprendizado de Máquina**](https://colab.research.google.com/github/Rogerio-mack/Inteligencia_Artificial/blob/main/IA_Python_1.ipynb) 

- Listas
- Dicionários

**3** [**Python Pandas e Matplotlib**](https://colab.research.google.com/github/Rogerio-mack/Inteligencia_Artificial/blob/main/IA_Python_2.ipynb)

- `Pandas`
  - Seleção de Dados: `df[ <predicado lógico> ][ <lista de colunas ]`, `tips[ df.tip > df.tip.mean() ][['total_bill','tip','sex']]`
  - `nlargest()` `nsmallest`
  - `pd.merge()`
  - `df.groupby()`

- `Matplotlib`
  - None!

**4** [**Aprendizado Supervisionado e Regressão Linear**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML2_Regressao.ipynb)

- Aprendizado Supervisionado $\rightarrow$ Conjunto de Treinamento, Dados rotulados
- **Esquema Geral dos Modelos Supervisionados**
- Regressão $\times$ |Classificação
- Regressão Linear Simples e Múltipla $y = a_0 + a_1 x_1 + ... + a_n x_n$
- Regressão Linear: Transformações da variável **dependente**, **OK**, já das **preditoras (ou independentes), NOK**
- Coeficientes $\rightarrow$ Minimização do Erro
- Coeficiente de Determinação, **$R2$**
  - R2 ajustado: None!
  - R2 $\in [0,1]$
  - p-value $< 0.05$ para os coeficientes
  - R2 = 0? Não há relação?
  - R2 = 1? Causa-Efeito? 
 - Variáveis Categóricas $\rightarrow$ Hot Encode  
 - Simple Code: `model = sm.ols('y ~ x',data=df; r = model.fit()`

**5** [**Classificação: Regressão Logística**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML3_RegressaoLogistica.ipynb)

- **Esquema Geral dos Modelos Supervisionados**
- Estimando os parâmetros, quais?
- **Esquema Geral dos Estimadores no `scikit-learn`**

- **Simples, sem X_train, X_test**
```
from sklearn.linear_model import LogisticRegression

# Entradas e Saídas
X = df[['radius_mean', 'texture_mean', 'perimeter_mean']]
y = df['diagnosis']

# Definição
clf = LogisticRegression()

# Treinamento
clf.fit(X,y)

# Avaliação
y_pred = clf.predict(X)

# Score do Modelo, Acuracidade
acc = clf.score(X,y)
```
- **Com X_train, X_test**
```
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

X = df[['x1','x2']]
y = df.y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=123)

clf = LogisticRegression(max_iter=1000)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print( clf.score(X_test,y_test) )
```
- Regressão Logística: **Classificador ou Separador Linear**
- Regressão Logística: **Classificador binário**
- Regressão Logística: variáveis preditoras numéricas, mas a dependente não necessariamente
- Dilema Viés-Variância: **Underfitting $\times$ Overfitting**
- Evitando Underfitting? 
- Evitando Overfitting? **Conjuntos de Treinamento e Teste**
- Acuracidade e Risco dessa métrica (classes desbalanceadas)
- *Um modelo de Deep Learning é sempre melhor que um modelo simples de Regressão Logística que só pode classificar corretamente dados linearmente separáveis.* True or False?

**6** [**Métricas de Classificação e K-Vizinhos mais Próximos**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML4_Knn.ipynb)

- Esquema geral do Knn
- **Matriz de Confusão, Precisão e Recall**

**7** **Proposta de Projeto, na data da prova**.

- Definição do Problema, Recursos e Referência (=uma atividade peso 2).

