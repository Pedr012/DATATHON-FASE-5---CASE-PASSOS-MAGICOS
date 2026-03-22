# 🎓 Sistema de Predição de Risco Educacional

## Associação Passos Mágicos - Tech Challenge Fase 5

### 📋 Sobre o Projeto

Este projeto foi desenvolvido como parte do Tech Challenge da Fase 5 da Pós-Graduação em Data Analytics da FIAP. O objetivo é analisar os dados educacionais da Associação Passos Mágicos e desenvolver um modelo preditivo para identificar alunos em risco de defasagem educacional, possibilitando intervenções precoces e personalizadas.

### 📁 Estrutura do Projeto

```
Tech Challenge/
├── Bases/
│   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
├── analise_passos_magicos.ipynb    # Análise completa e treinamento do modelo
├── app_streamlit.py                 # Aplicação web para predições
├── modelo_risco_defasagem.pkl       # Modelo treinado com pipeline completo
├── requirements.txt                 # Dependências Python
├── README.md                        # Este arquivo
└── GUIA_RAPIDO.md                   # Guia rápido de uso
```

---

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior
- Git (opcional)
- Navegador web moderno

### 1. Prepare o ambiente
```bash
# Navegue até a pasta do projeto
cd "Tech Challenge"

# Crie e ative o ambiente virtual (recomendado)
python -m venv TechFase5
TechFase5\Scripts\activate  # Windows
# source TechFase5/bin/activate  # Linux/Mac
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute o Notebook de Análise
```bash
# Inicie o Jupyter
jupyter notebook

# Abra o arquivo analise_passos_magicos.ipynb
# Execute todas as células para gerar o modelo
```

### 4. Execute a Aplicação Web
```bash
streamlit run app_streamlit.py

# A aplicação abrirá automaticamente em http://localhost:8501
```

---

## 📊 Modelo Preditivo

### Arquitetura
**Pipeline completo com pré-processamento e Gradient Boosting Classifier**

### Métricas de Performance (Conjunto de Teste)
- **ROC-AUC: 0.8230** - Boa capacidade de discriminação
- **Accuracy: 0.7457** - Taxa de acerto geral 74.6%
- **Precision: 0.7034** - Confiança nas predições positivas
- **Recall: 0.7757** - Captura 77.6% dos alunos em risco
- **F1-Score: 0.7378** - Bom equilíbrio entre precisão e recall

**Versão do Modelo:** 2.0
**Otimização:** Remoção de viés demográfico (IDADE removida)

**Features Mais Importantes:**
1. Número de Avaliações (NUM_AVALIACOES) - 14.9%
2. Indicador Psicopedagógico 2023 (IPP_2023) - 13.9%
3. Índice de Adequação ao Nível 2023 (IAN_2023) - 13.2%
4. Tempo no Programa (TEMPO_PROGRAMA) - 7.2%
5. Variância das Notas (VARIANCIA_NOTAS) - 6.8%

### Pipeline de Processamento

```python
1. Pré-processamento:
   - Label Encoding para variáveis categóricas (GENERO)
   - SimpleImputer para valores faltantes
   - StandardScaler para normalização

2. Engenharia de Features:
   - MEDIA_NOTAS: Média das três notas principais
   - VARIANCIA_NOTAS: Variabilidade do desempenho
   - NOTA_MIN / NOTA_MAX: Valores extremos
   - DESEMPENHO_EQUILIBRADO: Indicador de consistência
   - NOTAS_BAIXAS: Flag para desempenho crítico

3. Modelo:
   - Gradient Boosting Classifier
   - Otimizado para recall (prioriza detecção de risco)
   - Tuned para minimizar falsos negativos
```

**Features Utilizadas (Total: 27 variáveis - IDADE removida v2.0):**
- **Notas**: NOTA_PORT, NOTA_MAT, NOTA_ING
- **Aluno**: GENERO, TEMPO_PROGRAMA, NUM_AVALIACOES
- **Derivadas**: MEDIA_NOTAS, VARIANCIA_NOTAS, NOTA_MIN, NOTA_MAX, DESEMPENHO_EQUILIBRADO, NOTAS_BAIXAS
- **Históricas**: Indicadores de anos anteriores (2022 e 2023)

**⚠️ Por que IDADE foi removida?**
Análise de mediação demonstrou que a influência da idade no risco é 100% mediada pelas notas acadêmicas. Ou seja, idade afeta risco APENAS porque alunos iniciantes tendem a ter notas mais baixas. Removendo idade, evitamos viés demográfico mantendo capacidade preditiva.

---

## 💻 Aplicação Streamlit

A aplicação oferece uma interface profissional com 3 funcionalidades principais:

### 📝 1. Predição Individual
- Interface interativa com campos de entrada validados
- Entradas: **Notas (Português, Matemática, Inglês), Gênero, Tempo no Programa, Número de Avaliações**
- Visualização da probabilidade de risco com barra de progresso
- Classificação automática (Em Risco / Sem Risco)
- Recomendações baseadas no resultado
- ⚠️ **Nota:** Campo IDADE foi removido para evitar viés demográfico

### 📊 2. Predição em Lote
- Upload de arquivos CSV ou Excel com múltiplos alunos
- Validação automática de colunas obrigatórias
- Processamento em massa com barra de progresso
- Estatísticas do lote (total, em risco, risco médio)
- Exportação de resultados detalhados em CSV

### 🔬 3. Informações do Modelo
- Métricas de desempenho (ROC-AUC, Acurácia, Precisão, Recall, F1-Score)
- Top 10 variáveis mais importantes para predição
- Detalhes técnicos completos
- Lista de todas as features utilizadas

---

## 📈 Análises Realizadas no Notebook

O notebook `analise_passos_magicos.ipynb` contém análise completa estruturada em seções:

### 1. Preparação e Limpeza dos Dados
- Carregamento e merge de datasets
- Tratamento de valores faltantes
- Análise exploratória inicial

### 2. Análise Univariada e Multivariada
- Distribuições de variáveis numéricas e categóricas
- Correlações e relações entre indicadores
- Identificação de outliers

### 3. Engenharia de Features
- Criação de variáveis derivadas (médias, variâncias)
- Indicadores binários de risco
- Transformações e normalizações

### 4. Modelagem Preditiva
- Comparação de algoritmos (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Seleção do melhor modelo (Gradient Boosting)
- Otimização de hiperparâmetros

### 5. Avaliação de Desempenho
- Matriz de confusão
- Curva ROC
- Análise de feature importance
- Métricas detalhadas

### 6. Persistência do Modelo
- Salvamento do pipeline completo
- Artefato pronto para produção (modelo_risco_defasagem.pkl)

### Principais Análises Exploratórias
- **Adequação do nível (IAN)**: 98.6% dos alunos com alguma defasagem (22.2% severa, 76.4% moderada)
- **Desempenho Acadêmico (IDA)**: Evolução temporal e correlações
- **Engajamento (IEG)**: Correlação moderada com IDA (0.564)
- **Indicadores Multidimensionais**: IAA, IPS, IPP, IPV
- **Efetividade do Programa**: Distribuição por Pedras e taxas de Ponto de Virada por Fase

---

## 🎯 Principais Descobertas

1. **Desempenho do Modelo**: ROC-AUC de 0.8230 indica boa capacidade preditiva
2. **Recall de 77.6%**: Modelo identifica corretamente cerca de 3/4 dos alunos em risco
3. **Features mais relevantes**: Número de avaliações e indicadores históricos são os melhores preditores
4. **Otimização v2.0**: IDADE removida após análise de mediação demonstrar 100% de viés
5. **Engenharia de Features**: Variáveis derivadas das notas melhoram preditividade

---

## 💡 Recomendações Estratégicas

### 1. Sistema de Alerta Automatizado
- Implementar o modelo em produção para triagem contínua
- Alertas automáticos para casos de alto risco (probabilidade > 70%)

### 2. Intervenção Precoce e Personalizada
- Focar em alunos com média de notas abaixo de 6.0
- Priorizar casos com alta variância (inconsistência no desempenho)

### 3. Monitoramento de Indicadores-Chave
- Acompanhamento trimestral de MEDIA_NOTAS e VARIANCIA_NOTAS
- Dashboard com evolução temporal dos alunos em risco

### 4. Fortalecimento Pedagógico
- Reforço nas disciplinas com notas críticas (< 5.0)
- Estratégias para equilibrar desempenho entre matérias

### 5. Capacitação da Equipe
- Treinamento para interpretar predições do modelo
- Workshops sobre uso da aplicação Streamlit

### 6. Validação Contínua
- Retreinamento semestral com novos dados
- Monitoramento de drift nas predições

### 7. Engajamento Familiar
- Comunicação proativa com famílias de alunos em risco
- Planos de ação individualizados

---

## 🔧 Tecnologias Utilizadas

- **Python 3.10+**
- **Pandas & NumPy**: Manipulação e análise de dados
- **Scikit-learn 1.3+**: Machine Learning (Gradient Boosting, Pipeline, Preprocessamento)
- **Matplotlib & Seaborn**: Visualizações estatísticas
- **Plotly**: Visualizações interativas
- **Streamlit**: Interface web responsiva
- **Joblib**: Serialização eficiente do modelo
- **OpenPyXL**: Leitura de arquivos Excel

---

## 🌐 Deploy no Streamlit Community Cloud

### Passos para Deploy

#### 1. Crie um repositório no GitHub
- Faça upload de todos os arquivos do projeto
- Inclua: app_streamlit.py, requirements.txt, modelo_risco_defasagem.pkl

#### 2. Acesse Streamlit Community Cloud
- Vá para: https://streamlit.io/cloud
- Faça login com sua conta GitHub

#### 3. Deploy da Aplicação
- Clique em "New app"
- Selecione seu repositório
- Branch: main
- Main file path: app_streamlit.py
- Clique em "Deploy"

#### 4. Configurações Adicionais
- A aplicação será implantada automaticamente
- URL pública será gerada
- Atualizações automáticas ao fazer push no GitHub

---

## 📞 Contato

Projeto desenvolvido para o **Tech Challenge - Pós Tech FIAP**  
Fase 5: Deep Learning and Unstructured Data

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos.

---

**Associação Passos Mágicos** - Transformando vidas através da educação 🌟
