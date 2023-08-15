<p align="right">
  <img src="http://meusite.mackenzie.br/rogerio/mackenzie_logo/UPM.2_horizontal_vermelho.jpg" width="30%" align="center"/>
</p>

# Inteligência Artificial - Resumo N2

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

> None

**3** [**Python Pandas e Matplotlib**](https://colab.research.google.com/github/Rogerio-mack/Inteligencia_Artificial/blob/main/IA_Python_2.ipynb)

> None

**4** [**Aprendizado Supervisionado e Regressão Linear**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML2_Regressao.ipynb)

- Aprendizado Supervisionado $\rightarrow$ Conjunto de Treinamento, Dados rotulados
- **Esquema Geral dos Modelos Supervisionados**
- Regressão $\times$ |Classificação
- Regressão Linear Simples e Múltipla $y = a_0 + a_1 x_1 + ... + a_n x_n$

> None

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

**6** [**Métricas de Classificação e K-Vizinhos mais Próximos**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML4_Knn.ipynb)

- Conceito de K-Vizinhos mais Próximos
- Funções Distância de suas propriedades
- Necessidade de Normalização
- **Métricas** 
> **Classification Report**
>> Matriz de Confusão
>> Acuracidade
>> Precisão
>> Revocação (Recall)
>> F1-score
>> TP, TN, FP, FN

**7** [**Árvores de Decisão**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML5_DecisionTrees.ipynb)

- Conceitos de Árvores de Decisão
> Nó terminal
> Valores categóricos
> Método Partitivo
> Seleção dos nós raiz
- Entropia de valores notáveis
- Ganho de Informação

<br>

- **Seleção entre diferentes modelos: Regressão Logística, Knn, Árvore de Decisão, MLP**

<br>

**10** Redes Neurais

> [Introdução aos Modelos Neurais](https://colab.research.google.com/github/Rogerio-mack/Deep-Learning-I/blob/main/T1.ipynb)

> [MLP Modelo Multilayer Perceptron](https://colab.research.google.com/github/Rogerio-mack/Deep-Learning-I/blob/main/T2.ipynb)

- Conceitos de Redes Neurais
> Neurônio Perceptron
> Problema XOR
> Aprendizado e Backpropagation
> Ajuste dos pesos e funções de ativação
> Hiperparâmetros e parâmetros do modelo

**11** [Deep Learning](https://colab.research.google.com/github/Rogerio-mack/Deep-Learning-I/blob/main/T3.ipynb) ***(Conteúdo Opcional)***

> None

**12a** [Kmédias](https://colab.research.google.com/github/Rogerio-mack/BIG_DATA_Analytics_Mineracao_e_Analise_de_Dados/blob/main/BIG_T6_Clustering.ipynb) *Primeira parte do eBook*

- Conceito e tipos de Aprendizado não Supervisionado
> Clusterização, Detecção de Anomalia, Associação, Redução de Dimensionalidade
- Algoritmo de Kmédias
> Centróides
> Diferenças Clusterização $\times$ Clusterização
- Necessidade de Normalização
- Aplicação com o scikit-learn
- Seleção do K:
> Elbow
> Métrica de Silhouette
- Caracterização dos clusters

**12b** [Clusterização Hierárquica](https://colab.research.google.com/github/Rogerio-mack/BIG_DATA_Analytics_Mineracao_e_Analise_de_Dados/blob/main/BIG_T6_Clustering.ipynb) *Segunda parte do eBook*

- Conceito de Clusterização Hierárquica
> Agregação e Separação
> Matriz de Distância
> Dendograma
> Cortes para geração dos Cluster
> Diferenças com relação ao Kmédias
- Seleção do número de clusters:
> Métrica de Silhouette
- Caracterização dos clusters
