# MVP Bike Sharing

MVP de Machine Learning & Analytics para prever a demanda horaria de um sistema de bicicletas compartilhadas a partir de variaveis de calendario e clima.

O projeto foi desenvolvido como um relatorio tecnico executavel em notebook, com foco em um problema operacional real: antecipar a quantidade de bicicletas alugadas por hora para apoiar decisoes de rebalanceamento, dimensionamento de frota e planejamento da operacao.

## Visao geral

Sistemas de bike sharing precisam equilibrar oferta e demanda ao longo do dia. Bicicletas paradas em locais de baixa procura representam capital ocioso; estacoes vazias em locais de alta procura representam perda de receita e pior experiencia para o usuario.

Este MVP usa o Bike Sharing Dataset da UCI, com registros horarios do sistema Capital Bikeshare em Washington D.C. entre 2011 e 2012, para construir um modelo de regressao capaz de estimar a demanda horaria de alugueis.

## Objetivo

Construir, avaliar e interpretar um modelo preditivo que estime a variavel `cnt`, isto e, a contagem total de alugueis de bicicletas em uma determinada hora, usando atributos como:

- hora do dia;
- dia da semana;
- mes;
- feriado;
- dia util;
- estacao do ano;
- condicao do tempo;
- temperatura;
- umidade;
- velocidade do vento;
- ano.

A entrega concreta do MVP e uma funcao de inferencia, `prever_demanda(...)`, que recebe um cenario operacional e retorna a demanda estimada de bicicletas por hora.

## Pergunta analitica

Quao bem e possivel prever a contagem horaria de alugueis de bicicletas a partir de variaveis de calendario e clima, e quais fatores mais explicam essa demanda?

## Dados

Fonte: Bike Sharing Dataset, UCI Machine Learning Repository.

O repositorio inclui os arquivos:

- `dataset/hour.csv`: dados horarios, usados como base principal do MVP;
- `dataset/day.csv`: dados agregados por dia, mantidos como referencia do dataset original.

Resumo da base principal:

- periodo: 1 de janeiro de 2011 a 31 de dezembro de 2012;
- local: Washington D.C., EUA;
- granularidade: uma linha por hora;
- volume: 17.379 registros horarios;
- alvo: `cnt`, total de alugueis por hora.

## Metodologia

O notebook segue a estrutura CRISP-DM:

1. entendimento do problema de negocio;
2. entendimento dos dados;
3. preparacao dos dados;
4. modelagem;
5. avaliacao;
6. prototipo de uso.

As principais decisoes tecnicas foram:

- remocao de `casual` e `registered`, pois essas colunas compoem diretamente o alvo `cnt` e causariam vazamento;
- remocao de `atemp`, por alta colinearidade com `temp`;
- remocao de `instant` e `dteday`, por nao serem usadas diretamente como preditores;
- codificacao ciclica de `hr`, `mnth` e `weekday`;
- uso de `Pipeline` e `ColumnTransformer` para evitar vazamento de preprocessamento;
- comparacao entre baseline, Regressao Linear, Random Forest e XGBoost;
- otimizacao do XGBoost com `RandomizedSearchCV`;
- avaliacao complementar com split temporal para medir generalizacao futura.

## Modelos avaliados

Foram comparados:

- `DummyRegressor`, como baseline pela media historica;
- Regressao Linear, como modelo simples e interpretavel;
- Random Forest, como ensemble nao linear;
- XGBoost, como modelo final de gradient boosting para dados tabulares.

## Principais resultados

O melhor modelo foi o XGBoost otimizado.

Resultados aproximados em teste aleatorio:

- R2: cerca de 0,956;
- MAE: cerca de 23 bicicletas por hora;
- RMSE: cerca de 37 bicicletas por hora.

Comparacao com o baseline:

- baseline pela media historica: erro medio absoluto em torno de 140 bicicletas por hora;
- XGBoost otimizado: erro medio absoluto em torno de 23 bicicletas por hora.

Na avaliacao temporal, treinando no passado e testando no futuro, o desempenho recua para R2 em torno de 0,89. Esse resultado e mais realista para um cenario de producao e foi mantido como alerta metodologico sobre generalizacao temporal.

## Interpretabilidade

Os principais fatores identificados pelo modelo foram:

- hora do dia;
- dia util;
- ano, capturando o crescimento do sistema de 2011 para 2012;
- estacao do ano;
- temperatura;
- condicao do tempo.

Esses resultados sao coerentes com a analise exploratoria: a demanda tem picos claros de deslocamento diario, especialmente em dias uteis, e tambem varia com clima e sazonalidade.

## Estrutura do repositorio

```text
.
├── MVP_Bike_Sharing.ipynb
├── README.md
└── dataset
    ├── day.csv
    └── hour.csv
```

## Como executar

### Opcao 1: Google Colab

1. Abra o arquivo `MVP_Bike_Sharing.ipynb` no Google Colab.
2. Execute as celulas na ordem.
3. Caso necessario, o notebook instala dependencias como `xgboost` e `shap`.

### Opcao 2: ambiente local

Crie um ambiente Python e instale as bibliotecas principais:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
```

Depois, inicie o Jupyter:

```bash
jupyter notebook
```

Abra `MVP_Bike_Sharing.ipynb` e execute todas as celulas.

## Ferramentas utilizadas

- Python 3;
- pandas;
- NumPy;
- Matplotlib;
- Seaborn;
- scikit-learn;
- XGBoost;
- SHAP;
- Jupyter/Google Colab.

## Limitacoes

- Os dados representam apenas Washington D.C. em 2011-2012.
- O modelo nao deve ser aplicado diretamente a outra cidade ou periodo sem retreino.
- A base nao inclui eventos locais, chuva em milimetros, precos, promocoes ou disponibilidade real por estacao.
- Ha risco de data drift em aplicacoes reais.
- O modelo captura associacoes, nao causalidade.

## Proximos passos

- Usar validacao temporal mais rigorosa com `TimeSeriesSplit`.
- Adicionar features de defasagem e medias moveis.
- Incluir dados externos, como eventos, precipitacao e feriados locais.
- Modelar incerteza com intervalos de previsao.
- Empacotar o modelo com `joblib`.
- Expor a inferencia em uma API ou dashboard operacional.
- Monitorar drift e agendar retreinos periodicos.

## Referencias

- Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence.
- UCI Machine Learning Repository: Bike Sharing Dataset.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Lundberg, S., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
