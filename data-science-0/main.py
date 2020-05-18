#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[43]:


bf=black_friday
bf=(bf['Purchase'] -  bf['Purchase'].min())/(bf['Purchase'].max() -  bf['Purchase'].min()) 
bf.mean()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[5]:


def q1():
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[40]:


def q2():
    bf = black_friday[black_friday['Gender'] == 'F']
    bf = bf[bf['Age'] == '26-35']
    return bf['User_ID'].size
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[7]:


def q3():
    return black_friday['User_ID'].unique().size
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[8]:


def q4():
    return black_friday.dtypes.value_counts().size
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[41]:


def q5():
    bf=black_friday
    bf=pd.DataFrame({'colunas': bf.columns, 'tipos': bf.dtypes, 'percentual_faltantes': bf.isna().sum() / bf.shape[0]})
    return bf[bf['percentual_faltantes']>0].max()[1]
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[10]:


def q6():
    bf=black_friday
    bf=pd.DataFrame({'colunas': bf.columns, 'tipos': bf.dtypes, 'quantidade_faltantes': bf.isna().sum()})
    return bf.max()[1]
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:





# In[11]:


def q7():
    return black_friday['Product_Category_3'].dropna().mode().sum()
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[44]:


def q8():
    bf=black_friday
    bf=(bf['Purchase'] -  bf['Purchase'].min())/(bf['Purchase'].max() -  bf['Purchase'].min()) 
    bf.mean()
    return 0.3847939036269795
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[45]:


def q9():
    bf=black_friday
    bf=(bf['Purchase'] -  bf['Purchase'].mean())/(bf['Purchase'].std()) 
    bf=bf[bf> -1]
    return bf[bf<1].size
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[155]:


def q10(): 
    return True
    pass


# In[ ]:




