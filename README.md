# Datathon — Passos Mágicos
### Tech Challenge Fase 5 · FIAP Pós-Tech Data Analytics

Projeto de análise de dados educacionais e desenvolvimento de modelo preditivo de risco de defasagem escolar para a Associação Passos Mágicos, realizado como entrega do Tech Challenge da Fase 5 do programa de Pós-Graduação em Data Analytics da FIAP.

---

## Estrutura do Repositório

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── app_streamlit.py            # Aplicação web de predição (Streamlit)
│
├── notebooks/
│   └── analise_passos_magicos.ipynb   # Análise exploratória e treinamento do modelo
│
├── modelo/
│   ├── modelo_risco_defasagem.pkl     # Artefato principal (usado pelo app)
│   └── modelo_risco_educacional.pkl   # Versão de backup
│
├── data/
│   ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx   # Dataset principal
│   ├── Base de dados - Passos Magicos.zip         # Dados históricos
│   └── referencias/
│       ├── Dicionario Dados Datathon.pdf
│       ├── desvendando_passos.pdf
│       ├── Relatorio PEDE2020.pdf
│       ├── Relatorio PEDE2021.pdf
│       └── Relatorio PEDE2022.pdf
│
└── docs/
    ├── relatorio_passos_magicos.docx
    ├── DATATHON - Case Passos Magicos.docx
    ├── POSTECH - DTAT - Datathon - Fase 5.pdf
    ├── Links adicionais da passos.docx
    └── PEDE_ Pontos importantes.docx
```

---

## Contexto do Problema

A Associação Passos Mágicos atua no desenvolvimento educacional de crianças e jovens em situação de vulnerabilidade. O programa PEDE (Pesquisa de Desenvolvimento Educacional) coleta anualmente indicadores acadêmicos e socioeducacionais dos alunos.

O desafio proposto consiste em identificar, com base nesses indicadores, quais alunos estão em risco de defasagem educacional — permitindo intervenção precoce e personalizada pela equipe pedagógica.

---

## Solução Desenvolvida

### Análise Exploratória (notebook)

O notebook `notebooks/analise_passos_magicos.ipynb` cobre:

- Carga e merge dos datasets PEDE 2022, 2023 e 2024
- Padronização e tratamento de valores ausentes
- Análise univariada e multivariada dos indicadores (IAN, IDA, IEG, IPP, IAA, IPS, IPV, INDE)
- Engenharia de features (médias, variâncias, flags de risco)
- Comparação de algoritmos: Regressão Logística, Árvore de Decisão, Random Forest, Gradient Boosting
- Seleção, tunagem e avaliação final do modelo
- Serialização do artefato para produção

### Modelo Preditivo

**Algoritmo selecionado:** Gradient Boosting Classifier  
**Pipeline:** padronização + imputação + encoding + modelo calibrado

**Desempenho no conjunto de teste:**

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.8230 |
| Acurácia | 74.57% |
| Precisão | 70.34% |
| Recall | 77.57% |
| F1-Score | 73.78% |

**Principais variáveis preditoras:**
1. Número de avaliações (NUM_AVALIACOES) — 14.9%
2. Indicador Psicopedagógico 2023 (IPP_2023) — 13.9%
3. Índice de Adequação ao Nível 2023 (IAN_2023) — 13.2%
4. Tempo no programa (TEMPO_PROGRAMA) — 7.2%
5. Variância das notas (VARIANCIA_NOTAS) — 6.8%

> **Nota sobre IDADE:** o campo foi removido na versão 2.0 após análise de mediação demonstrar que sua influência no risco é 100% mediada pelas notas — ou seja, incluí-la introduzia viés demográfico sem ganho preditivo real.

### Aplicação Web (Streamlit)

`app_streamlit.py` oferece três funcionalidades:

- **Predição individual:** formulário com notas (Português, Matemática, Inglês), gênero, tempo no programa e número de avaliações. Retorna probabilidade de risco, classificação e recomendações.
- **Predição em lote:** upload de arquivo CSV ou Excel com múltiplos alunos, validação de colunas, estatísticas do lote e exportação de resultados.
- **Informações do modelo:** métricas de desempenho, importância das features e detalhes técnicos.

---

## Como Executar

### Pré-requisitos

- Python 3.10 ou superior

### Instalação

```bash
# Clone o repositório
git clone https://github.com/Pedr012/DATATHON-FASE-5---CASE-PASSOS-MAGICOS.git
cd DATATHON-FASE-5---CASE-PASSOS-MAGICOS

# Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Instale as dependências
pip install -r requirements.txt
```

### Executar a aplicação

```bash
streamlit run app_streamlit.py
```

A aplicação abrirá em `http://localhost:8501`.

### Executar o notebook

```bash
jupyter notebook notebooks/analise_passos_magicos.ipynb
```

> O notebook lê o dataset de `data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx` e salva o modelo em `modelo/modelo_risco_defasagem.pkl`. Execute a partir da raiz do repositório.

---

## Deploy no Streamlit Cloud

1. Faça fork ou push do repositório para o GitHub
2. Acesse [streamlit.io/cloud](https://streamlit.io/cloud) e conecte sua conta
3. Crie um novo app apontando para este repositório
4. **Main file path:** `app_streamlit.py`
5. Clique em *Deploy*

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| pandas / numpy | Manipulação de dados |
| scikit-learn | Pipeline, modelo, pré-processamento |
| xgboost | Gradient Boosting |
| matplotlib / seaborn / plotly | Visualizações |
| streamlit | Interface web |
| joblib | Serialização do modelo |
| openpyxl | Leitura de arquivos Excel |

---

## Equipe

Projeto desenvolvido para o **Tech Challenge — Pós-Tech FIAP**  
Fase 5: Deep Learning and Unstructured Data