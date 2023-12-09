import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

df = pd.read_csv('Clean Data_pakwheels.csv', sep=',')
df = df.drop(df.columns[0], axis=1)
df_filtrado = df[df.groupby('Model Name')['Model Name'].transform('size') >= 20]
df_filtrado = df_filtrado[df_filtrado['Engine Capacity'] != 16]

st.title("Dashboard Modelo de Previsão de Preços")

st.subheader("Dados Faltantes")
st.write(f'Dados Faltantes: {df.isnull().any().any()}')

st.write(f"Quantidade de modelos antes do fitro: {df['Model Name'].nunique()}")
st.write(f"Quantidade de modelos após o fitro: {df_filtrado['Model Name'].nunique()}")
st.write("Modelos com poucos dados foram retirados, pois não são dados confiáveis e Engine Capacity 16, por não fazer sentido para um veículo")


st.subheader("Tabela Descritiva Antes do Filtro de Modelos")
st.dataframe(df.describe())

st.subheader("Tabela Descritiva Depois do Filtro de Modelos")
st.dataframe(df_filtrado.describe())

st.subheader("Boxplot Preços Antes do Filtro de Modelos")
fig_antes, ax_antes = plt.subplots()
ax_antes.boxplot(df['Price'])
st.pyplot(fig_antes)

st.subheader("Boxplot Preços Depois do Filtro de Modelos")
fig_depois, ax_depois = plt.subplots()
ax_depois.boxplot(df_filtrado['Price'])
st.pyplot(fig_depois)

st.subheader("Histograma de Capacidade do Motor")
fig_capacidade_motor, ax_capacidade_motor = plt.subplots()
ax_capacidade_motor.hist(df_filtrado['Engine Capacity'], bins=10, color='blue', edgecolor='black')
ax_capacidade_motor.set_xlabel('Capacidade do Motor')
ax_capacidade_motor.set_ylabel('Frequência')
st.pyplot(fig_capacidade_motor)

st.subheader("Histograma do Preço")
fig_preco, ax_preco = plt.subplots()
ax_preco.hist(df_filtrado['Price'], bins=100, color='blue', edgecolor='black')
ax_preco.set_xlabel('Preço')
ax_preco.set_ylabel('Frequência')
st.pyplot(fig_preco)

st.subheader("Histograma da Quilometragem")
fig_km, ax_km = plt.subplots()
ax_km.hist(df_filtrado['Mileage'], bins=20, color='blue', edgecolor='black')
ax_km.set_xlabel('Quilometragem')
ax_km.set_ylabel('Frequência')
st.pyplot(fig_km)

colunas = ['Price', 'Model Year', 'Mileage', 'Engine Capacity']
df_colunas_para_corr = df_filtrado[colunas]
st.subheader("Mapa para Correlação")
fig_corr = sns.clustermap(df_colunas_para_corr.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True)
st.pyplot(fig_corr.fig)


df_escalonado = df_filtrado.copy()
scaler = StandardScaler()
df_escalonado['Price'] = scaler.fit_transform(df_escalonado[['Price']])
df_escalonado['Mileage'] = scaler.fit_transform(df_escalonado[['Mileage']])
df_escalonado['Engine Capacity'] = scaler.fit_transform(df_escalonado[['Engine Capacity']])




caracteristicas = ['Model Year', 'Mileage', 'Engine Capacity']
alvo = 'Price'

X = df_escalonado[caracteristicas]
y = df_escalonado[alvo]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()

modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)


coeficientes = modelo.coef_
coeficientes_str = ', '.join([f'{coef:.2f}' for coef in coeficientes])
st.subheader("Coeficientes do Modelo")
for nome, coef in zip(caracteristicas, coeficientes):
    st.write(f'{nome}: {coef:.2f}')


mse = mean_squared_error(y_teste, previsoes)
rmse = mean_squared_error(y_teste, previsoes, squared=False)  
mae = mean_absolute_error(y_teste, previsoes)

st.subheader("Métricas do Modelo")
st.write(f'Erro Quadrático Médio (MSE): {mse:.2f}')
st.write(f'Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}')
st.write(f'Erro Absoluto Médio (MAE): {mae:.2f}')



