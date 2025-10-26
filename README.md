# Manutenção Preditiva (PdM) com LSTM e FastAPI

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blueviolet?logo=scikit-learn)

## 1. Visão Geral do Projeto

Este é um projeto **end-to-end** de Machine Learning focado em **Manutenção Preditiva (PdM)**. O objetivo é prever o "Tempo Restante de Vida Útil" (Remaining Useful Life - RUL) de motores de avião, com base em dados de sensores de séries temporais.

O projeto utiliza o renomado dataset [NASA Turbofan Engine Degradation (CMAPS)](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) para treinar um modelo de Deep Learning (LSTM). O modelo treinado é então "servido" através de uma **API REST de alta performance construída com FastAPI**, capaz de receber dados de sensores em tempo real e retornar uma previsão de RUL.

## 2. O Problema de Negócio

Em ambientes industriais e aeronáuticos, falhas de equipamento não programadas (paradas, ou *downtime*) geram custos imensos, tanto em reparos quanto em perda de operação.

* **Manutenção Corretiva (Reativa):** Consertar *depois* que quebra. É o cenário mais caro.
* **Manutenção Preventiva (Baseada em tempo):** Trocar peças em intervalos fixos (ex: a cada 1000 horas). É seguro, mas ineficiente, pois muitas peças são descartadas ainda com vida útil.
* **Manutenção Preditiva (PdM):** O "padrão-ouro". Usar dados para prever *quando* uma falha está prestes a ocorrer, permitindo que a manutenção seja agendada no momento exato: nem muito cedo, nem tarde demais.

Este projeto resolve o problema da PdM, respondendo à pergunta: **"Quantos ciclos operacionais este motor ainda tem antes de falhar?"**

## 3. A Solução de Machine Learning

Para resolver este problema de previsão em séries temporais, foi implementado o seguinte pipeline:

1.  **Engenharia de Features:** O dataset de treino (`train_FD001.txt`) contém dados de 100 motores operados até a falha. A coluna alvo `RUL` foi criada calculando a diferença entre o ciclo de falha de cada motor e seu ciclo atual.
2.  **Pré-processamento:**
    * Features irrelevantes (sensores com variação zero) foram removidas.
    * Os dados dos sensores foram normalizados usando `MinMaxScaler` (salvo em `scaler.pkl`) para otimizar a convergência da rede neural.
3.  **Sequenciamento (Janelas Deslizantes):** Modelos LSTM não analisam um ciclo de cada vez; eles analisam uma *sequência* de ciclos. Os dados foram transformados em "janelas" de 50 ciclos (`sequence_length = 50`) para prever o RUL no final dessa janela.
4.  **Modelagem (Deep Learning):** Uma Rede Neural Recorrente (RNN) do tipo **LSTM (Long Short-Term Memory)** foi construída e treinada. A arquitetura empilhada (100 unidades -> 50 unidades) foi escolhida por sua capacidade de "lembrar" padrões de longo prazo na degradação dos sensores.
5.  **Deployment:** O modelo treinado (`.h5`) e o scaler (`.pkl`) são carregados por uma API FastAPI, que expõe um endpoint `/predict` para previsões em tempo real.

## 4. Tech Stack (Tecnologias Utilizadas)

* **Análise e Modelagem:** Python 3.11+, Jupyter Notebooks
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (para pré-processamento e métricas)
* **Deep Learning:** TensorFlow (Keras) (para a arquitetura LSTM)
* **Servidor da API:** FastAPI, Uvicorn
* **Serialização:** Joblib (para salvar o `scaler`)

## 5. Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
[ nome_do_projeto_pdm/ ]
|
|-- api/
|   |-- app.py                  <-- (Lógica do servidor FastAPI)
|   |-- requirements_api.txt    <-- (Dependências da API)
|
|-- data/
|   |-- raw/
|   |   |-- train_FD001.txt
|   |   |-- test_FD001.txt
|   |   |-- RUL_FD001.txt
|   |
|   |-- processed/
|       |-- train_FD001_processed.csv  <-- (Gerado pelo notebook)
|
|-- models/
|   |-- rul_predictor_lstm.h5   <-- (Modelo treinado, gerado pelo notebook)
|   |-- scaler.pkl              <-- (Scaler treinado, gerado pelo notebook)
|
|-- notebooks/
|   |-- PdM_LSTM_Treinamento_Completo.ipynb  <-- (Notebook com os Passos 1-5)
|
|-- requirements.txt            <-- (Dependências do notebook de treino)
|-- README.md                   <-- (Este arquivo)
```

## 6. Como Usar

### Pré-requisitos

* Python 3.11 ou superior
* Um ambiente virtual (ex: `.venv`) é recomendado.

### Instalação

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/seu-usuario/nome_do_projeto_pdm.git](https://github.com/seu-usuario/nome_do_projeto_pdm.git)
    cd nome_do_projeto_pdm
    ```

2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  Instale as dependências de treinamento:
    ```bash
    pip install -r requirements.txt
    ```

4.  Instale as dependências da API:
    ```bash
    pip install -r api/requirements_api.txt
    ```

### Treinamento (Opcional - Modelo já treinado)

Para treinar o modelo do zero:

1.  Abra o Jupyter Notebook:
    ```bash
    jupyter notebook notebooks/PdM_LSTM_Treinamento_Completo.ipynb
    ```
2.  No VS Code, apenas abra o arquivo `.ipynb`.
3.  Execute todas as células em ordem. Isso irá recriar os arquivos em `data/processed/` e `models/`.

### Executando a API (Servidor Local)

Com os arquivos `rul_predictor_lstm.h5` e `scaler.pkl` na pasta `models/`, execute o servidor FastAPI a partir da pasta raiz do projeto:

```bash
uvicorn api.app:app --reload
```

O servidor estará disponível em `http://127.0.0.1:8000`.

### Testando a API

Você pode acessar `http://127.0.0.1:8000/docs` no seu navegador para ver a documentação interativa do Swagger UI.

Alternativamente, use o `curl` em um novo terminal para enviar uma requisição de previsão (exemplo com os dados de 1 ciclo e 19 features):

```bash
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
-H "Content-Type: application/json" \
-d "{\"cycles\": [[-0.0007, -0.0004, 641.82, 1589.7, 1400.6, 14.62, 21.61, 554.36, 2388.06, 9046.19, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 39.06]]}"
```

A API responderá com o RUL previsto:
```json
{
  "predicted_RUL": 168.45
}
```
*(O valor exato pode variar ligeiramente.)*

## 7. Resultados do Modelo

O modelo foi avaliado no conjunto de teste (`test_FD001.txt`) e comparado com os valores reais (`RUL_FD001.txt`).

* **RMSE (Root Mean Squared Error): 26.06 ciclos**
* **MAE (Mean Absolute Error): 17.70 ciclos**

**Interpretação:** Em média, o modelo consegue prever o tempo restante de vida útil com um erro de aproximadamente **18 ciclos**.

O gráfico de dispersão abaixo mostra a forte correlação entre os valores reais (eixo X) e as previsões do modelo (eixo Y).

<Figure size 1200x600 with 1 Axes><img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/ae8af9a0-d9bf-4728-915e-1c84abc9ebfc" />

