import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Carregar datasets
tourism_data = pd.read_csv("dataset/tax-sales-hurricane.csv", parse_dates=["Date"])
employment_data = pd.read_csv("dataset/Employment.csv", parse_dates=["Date"])
covid_data = pd.read_csv("dataset/covid-19.csv", parse_dates=["Date"])


# Pré-processamento
def preprocess_data(df, key_columns):
    df = df[key_columns].dropna()
    return df


tourism_data = preprocess_data(
    tourism_data, ["Date", "region", "observed", "hurricane"]
)
covid_data = preprocess_data(covid_data, ["Date", "stateAbbr", "dailynewcases"])
employment_data = preprocess_data(employment_data, ["Date", "stateFull", "Employees"])

# Unificar datasets com base na data
merged_data = tourism_data.merge(
    covid_data, left_on="Date", right_on="Date", how="left"
)
merged_data = merged_data.merge(
    employment_data, left_on="Date", right_on="Date", how="left"
)

# Normalizar os dados
scaler = MinMaxScaler()
merged_data["observed_scaled"] = scaler.fit_transform(merged_data[["observed"]])


# Preparar dados para modelagem
def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


sequence_length = 12  # Dados mensais, janela de 1 ano
observed_values = merged_data["observed_scaled"].values
X, y = prepare_sequences(observed_values, sequence_length)

# Divisão em treino e teste
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Reformatar para o formato esperado pelo LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Construir o modelo LSTM
model = Sequential(
    [
        LSTM(
            50,
            activation="relu",
            input_shape=(sequence_length, 1),
            return_sequences=True,
        ),
        Dropout(0.2),
        LSTM(50, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Fazer previsões
predictions = model.predict(X_test)

# Reverter a normalização
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions = scaler.inverse_transform(predictions)

# Visualizar os resultados
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label="Real", color="blue")
plt.plot(range(len(predictions)), predictions, label="Previsto", color="orange")
plt.title("Previsão de Receitas com Impacto de Furacões", fontsize=14)
plt.xlabel("Tempo", fontsize=12)
plt.ylabel("Receita (Milhões de Dólares)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(visible=True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Avaliar o modelo
mse = np.mean((y_test - predictions) ** 2)
print(f"Mean Squared Error: {mse:.2f}")

plt.figure(figsize=(14, 7))
plt.plot(range(len(y_test)), y_test, label="Real", color="blue")
plt.plot(range(len(predictions)), predictions, label="Previsto", color="orange")
plt.title("Previsão x Real (Zoom)")
plt.xlabel("Tempo")
plt.ylabel("Receita")
plt.legend()
plt.xlim(0, 50)  # Ajustar para zoom no período inicial de teste
plt.show()

outliers = merged_data[
    (merged_data["observed"] > merged_data["observed"].quantile(0.95))
    | (merged_data["observed"] < merged_data["observed"].quantile(0.05))
]
plt.figure(figsize=(14, 7))
plt.plot(merged_data["Date"], merged_data["observed"], label="Receitas Observadas")
plt.scatter(
    outliers["Date"],
    outliers["observed"],
    color="red",
    label="Outliers",
    alpha=0.7,
)
plt.title("Detecção de Outliers na Série Temporal")
plt.xlabel("Tempo")
plt.ylabel("Receitas (Milhões de Dólares)")
plt.legend()
plt.show()
