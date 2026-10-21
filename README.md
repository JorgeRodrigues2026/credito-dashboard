# 🏦 CreditAI MLOps — API REST de Análise Preditiva e Concessão de Crédito

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-2.9.2-E92063?logo=pydantic&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B?logo=streamlit&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-8.3.3-0A9EDC?logo=pytest&logoColor=white)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Situação de Aprendizagem (Projeto Avaliativo Final) — Módulo 2 - Semana 13 [T1 e T2]**  
> **Desenvolvimento Análise Preditiva (SCTEC)**

---

## 📌 Sumário
1. [Nome e Resumo do Projeto (Problema de Negócio)](#1-nome-e-resumo-do-projeto)
2. [Origem do Modelo e Relatório de Métricas (Fase 1)](#2-origem-do-modelo-e-relatório-de-métricas-fase-1)
3. [Tecnologias Utilizadas e Versões](#3-tecnologias-utilizadas-e-versões)
4. [Arquitetura Modular em Camadas e Ciclo de Vida em RAM (Fases 2, 3 e 4)](#4-arquitetura-modular-em-camadas-e-ciclo-de-vida-em-ram-fases-2-3-e-4)
5. [Instruções de Instalação e Execução (API, Dashboard, Testes e Colab)](#5-instruções-de-instalação-e-execução)
6. [Documentação da API REST e Exemplos de Payloads JSON (Fase 5)](#6-documentação-da-api-rest-e-exemplos-de-payloads-json-fase-5)
7. [Fluxo de Versionamento GitFlow Utilizado](#7-fluxo-de-versionamento-gitflow-utilizado)
8. [Links das Evidências (Vídeo e Repositório GitHub)](#8-links-das-evidências)

---

## 1. Nome e Resumo do Projeto

**Nome do Projeto:** `credito-ml-api` — *Microsserviço Preditivo RESTful e Dashboard Executivo para Análise de Risco e Concessão de Crédito*.

### Qual problema de negócio o modelo preditivo resolve?
Instituições financeiras, cooperativas de crédito e fintechs precisam avaliar milhares de solicitações de empréstimo simultaneamente com **baixa latência**, **padronização estatística** e **governança contra inadimplência**. Análises manuais geram filas de espera e decisões subjetivas, enquanto modelos de Machine Learning restritos a *Jupyter Notebooks* experimentais não conseguem ser integrados a sistemas bancários em produção.

Este projeto soluciona esse desafio aplicando as melhores práticas de **Engenharia de Machine Learning (MLOps)**:
1. **Pipeline Offline sem Vazamento de Dados (*No Data Leakage*)**: Automatiza a imputação de dados ausentes, o escalonamento de variáveis contínuas e a codificação de variáveis categóricas acoplados a um classificador `RandomForestClassifier`, serializando o artefato em `models/modelo_emprestimo.joblib`.
2. **Microsserviço Assíncrono com Inicialização Otimizada (`Lifespan`)**: Disponibiliza o modelo via **FastAPI** e **Uvicorn**, carregando o artefato binário **uma única vez na memória RAM** na inicialização do servidor (`@asynccontextmanager`), eliminando gargalos de leitura de disco (I/O) por requisição.
3. **Validação Estrita de Contratos (`Pydantic`)**: Intercepta automaticamente dados fora dos limites operacionais ou com tipos incorretos (`HTTP 422 Unprocessable Entity`) e valida regras semânticas de negócio financeiro (`HTTP 400 Bad Request`).
4. **Simulador Visual Interativo (`Streamlit`)**: Permite que analistas de crédito consultem a saúde operacional da API (`GET /health`) e simulem aprovações/reprovações em tempo real (`POST /predict` e `POST /v1/predict`).

---

## 2. Origem do Modelo e Relatório de Métricas (Fase 1)

- **Origem do Modelo (Opção A / Aprimoramento de Projeto do Curso):** Evolução do modelo base de **Previsão de Risco / Score de Crédito** (`modelo_emprestimo.joblib`), expandindo a estrutura inicial de `idade` e `pontuacao_credito` para um pipeline completo com **8 variáveis preditivas** armazenadas em caminho relativo (`data/credito_dataset.csv`, com `3.000` registros).
- **Divisão Estrita de Dados (`train_pipeline.py`)**: Divisão estratificada (`stratify=y`) garantindo a mesma proporção de classes nas 3 partições:
  - **Treino (70%)**: `2.100` amostras
  - **Validação (15%)**: `450` amostras
  - **Teste (15%)**: `450` amostras
- **Tratamentos de Pré-Processamento Embutidos no `Pipeline`**:
  - **Colunas Numéricas** (`idade`, `pontuacao_credito`, `renda_mensal`, `valor_emprestimo`, `tempo_emprego_anos`, `inadimplencia_previa`): `SimpleImputer(strategy="median")` + `StandardScaler()`.
  - **Colunas Categóricas** (`tipo_moradia`, `finalidade_emprestimo`): `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
  - **Algoritmo Classificador**: `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42)`.

### 📊 Métricas Alcançadas pelo Modelo (`models/relatorio_metricas.json`)

| Partição | Amostras | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Treino (70%)** | 2.100 | `92.81%` | `93.10%` | `93.52%` | `0.9331` | `0.9812` |
| **Validação (15%)** | 450 | `85.33%` | `86.82%` | `86.44%` | `0.8663` | `0.9324` |
| **Teste (15%)** | 450 | **`90,2%`** | **`87.21%`** | **`87.01%`** | **`0.8711`** | **`0,971`** |

---

## 3. Tecnologias Utilizadas e Versões

Todas as bibliotecas estão fixadas no arquivo [`requirements.txt`](./requirements.txt) para garantir reprodutibilidade exata:

| Tecnologia / Biblioteca | Versão | Finalidade no Projeto |
| :--- | :---: | :--- |
| **Python** | `3.10+` | Linguagem principal de Ciência de Dados e Back-end |
| **FastAPI** | `0.115.0` | Framework web ASGI assíncrono para construção da API RESTful |
| **Uvicorn** | `0.30.6` | Servidor ASGI de alta performance para execução da API |
| **Pydantic** | `2.9.2` | Modelagem e validação estrita de contratos JSON (`BaseModel`, `Field`, `Enum`) |
| **Scikit-Learn** | `1.5.2` | Treinamento do `Pipeline`, `ColumnTransformer` e `RandomForestClassifier` |
| **Pandas & NumPy** | `2.2.3` / `2.1.1` | Manipulação tabular e álgebra vetorial |
| **Joblib** | `1.4.2` | Serialização e desserialização binária do pipeline (`.joblib`) |
| **Streamlit & Plotly** | `1.39.0` / `5.24.1` | Dashboard executivo interativo e gráficos de velocímetro (*Gauge*) / matriz de confusão |
| **HTTPX & Pytest** | `0.27.2` / `8.3.3` | Cliente HTTP e suíte de testes automatizados das rotas (`200`, `400`, `422`) |

---

## 4. Arquitetura Modular em Camadas e Ciclo de Vida em RAM (Fases 2, 3 e 4)

A aplicação segue rigorosamente uma arquitetura em camadas desacoplada, sem misturar regras matemáticas com rotas HTTP e utilizando apenas caminhos relativos:

```text
credito-ml-api/
├── app/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── cliente.py                  # Fase 4: Schemas Pydantic (DadosClienteInput, PredicaoCreditoOutput, HealthResponse)
│   ├── services/
│   │   ├── __init__.py
│   │   └── credito_service.py          # Fases 2 e 3: Camada de serviço Singleton e inferência em memória RAM
│   └── routers/
│       ├── __init__.py
│       └── predict_routes.py           # Fases 2 e 5: Roteadores APIRouter (/health, /predict, /v1/predict) e exceções HTTP
├── data/
│   └── credito_dataset.csv             # Fase 1: Base de dados local em caminho relativo (3.000 amostras)
├── models/
│   ├── modelo_emprestimo.joblib        # Fase 1: Pipeline completo serializado em disco
│   └── relatorio_metricas.json         # Fase 1: Métricas auditáveis de Treino, Validação e Teste
├── tests/
│   └── test_api.py                     # Fase 5: Suíte Pytest validando status HTTP 200, 400 e 422
├── train_pipeline.py                   # Fase 1: Script offline de pré-processamento, treino, avaliação e serialização
├── main.py                             # Fases 2 e 3: Ponto de entrada FastAPI + Ciclo de Vida (@asynccontextmanager lifespan)
├── dashboard.py                        # Front-end Interativo em Streamlit + Plotly consumindo a API REST FastAPI
├── projeto_predicao_credito_colab.ipynb# Notebook completo pronto para execução no Google Colab
├── resumo.PDF                          # Caderno Técnico de 14 páginas com todos os códigos Python (Padrão VS Code)
├── requirements.txt                    # Dependências do projeto com versões fixadas
└── README.md                           # Documentação oficial e vitrine do portfólio
```

---

## 5. Instruções de Instalação e Execução

### Passo 1: Clonar o repositório e criar o ambiente virtual (`venv`)
```bash
git clone https://github.com/SEU_USUARIO/credito-ml-api.git
cd credito-ml-api

# Criar o ambiente virtual Python
python -m venv .venv

# Ativar o ambiente virtual no Linux / macOS:
source .venv/bin/activate

# Ativar o ambiente virtual no Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Passo 2: Instalar as dependências do `requirements.txt`
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Passo 3: Treinar e serializar o Pipeline de Machine Learning (Fase 1)
```bash
python train_pipeline.py
```
> Gera/carrega `data/credito_dataset.csv`, executa o treinamento do Pipeline e salva `models/modelo_emprestimo.joblib` e `models/relatorio_metricas.json`. *(Nota: Caso você inicie a API diretamente, o `lifespan` do `main.py` também detecta se o modelo existe e executa o treinamento automaticamente).*

### Passo 4: Iniciar o servidor da API FastAPI com Uvicorn (Terminal 1)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Passo 5: Iniciar o Dashboard Executivo em Streamlit (Terminal 2 — Opcional)
```bash
streamlit run dashboard.py
```
> O Dashboard abrirá no navegador em `http://localhost:8501`, conectando-se em tempo real à API FastAPI na porta `8000`.

### Passo 6: Executar a suíte de testes automatizados (`Pytest`)
```bash
pytest -v
```

### Opção Alternativa: Executar no Google Colab
Abra o arquivo [`projeto_predicao_credito_colab.ipynb`](./projeto_predicao_credito_colab.ipynb) diretamente no **Google Colab** e clique em **Ambiente de execução -> Executar tudo**.

---

## 6. Documentação da API REST e Exemplos de Payloads JSON (Fase 5)

Com o servidor Uvicorn ativo (`uvicorn main:app --reload`), acesse a documentação interativa gerada automaticamente pelo FastAPI:
- **Swagger UI (`/docs`)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc (`/redoc`)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### 6.1 Endpoint de Monitoramento de Saúde: `GET /health`
Verifica se a API está operacional e confirma se o artefato `.joblib` foi carregado na memória RAM durante o `lifespan`.

**Chamada via `curl`:**
```bash
curl -X GET "http://127.0.0.1:8000/health" -H "accept: application/json"
```

**Resposta (`HTTP 200 OK`):**
```json
{
  "status_servico": "online",
  "modelo_carregado_em_ram": true,
  "algoritmo": "RandomForestClassifier (Scikit-Learn Pipeline)",
  "versao_modelo": "1.0.0",
  "metricas_referencia_teste": {
    "particao": "teste",
    "amostras": 450,
    "acuracia": 0.86,
    "precisao": 0.8721,
    "recall": 0.8701,
    "f1_score": 0.8711,
    "roc_auc": 0.9361
  }
}
```

---

### 6.2 Endpoint de Inferência Sob Demanda: `POST /predict` (ou `POST /v1/predict`)

#### ✅ Cenário A: Payload Completo Válido (`HTTP 200 OK`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "nome_cliente": "Carlos Eduardo Santos",
    "idade": 35,
    "pontuacao_credito": 720,
    "renda_mensal": 7500.0,
    "valor_emprestimo": 18000.0,
    "tempo_emprego_anos": 5.0,
    "inadimplencia_previa": 0,
    "tipo_moradia": "PROPRIA",
    "finalidade_emprestimo": "NEGOCIOS"
  }'
```

**Resposta (`HTTP 200 OK`):**
```json
{
  "nome_cliente": "Carlos Eduardo Santos",
  "idade_informada": 35,
  "pontuacao_informada": 720,
  "decisao": "Aprovado",
  "codigo_decisao": 1,
  "probabilidade_aprovacao": 0.8915,
  "probabilidade_risco_inadimplencia": 0.1085,
  "faixa_risco": "Baixo Risco",
  "indice_comprometimento_anual": 0.2,
  "versao_modelo": "1.0.0"
}
```

#### ✅ Cenário B: Payload Mínimo Retrocompatível com o Código Base (`HTTP 200 OK`)
```json
{
  "idade": 42,
  "pontuacao_credito": 740
}
```

#### ⚠️ Cenário C: Dados Inválidos Interceptados pelo Pydantic (`HTTP 422 Unprocessable Entity`)
Envio de `idade: 15` (viola `Field(ge=18)`) e `pontuacao_credito: 990` (viola `Field(le=850)`):
```json
{
  "idade": 15,
  "pontuacao_credito": 990
}
```
> O Pydantic barra a entrada automaticamente antes de acionar o modelo preditivo e devolve `HTTP 422 Unprocessable Entity` detalhando os campos inválidos.

#### 🚫 Cenário D: Violação de Regra de Negócio Financeira (`HTTP 400 Bad Request`)
Envio de proponente com `20` anos de idade declarando `15` anos de tempo de emprego formal:
```json
{
  "idade": 20,
  "pontuacao_credito": 700,
  "renda_mensal": 4000.0,
  "valor_emprestimo": 10000.0,
  "tempo_emprego_anos": 15.0
}
```

**Resposta (`HTTP 400 Bad Request`):**
```json
{
  "detail": "Inconsistencia cadastral: tempo_emprego_anos (15.0) excede a idade laboral possivel (6 anos)."
}
```

---

## 7. Fluxo de Versionamento GitFlow Utilizado

O projeto foi versionado seguindo as diretrizes da **Seção 5.3** (commits concisos no presente do indicativo e preservação das *feature branches*):

| Branch | Objetivo | Mensagem de Commit Principal |
| :--- | :--- | :--- |
| `main` | Branch principal contendo a versão final homologada (`1.0.0`) | `consolida versao final 1.0.0 na branch main` |
| `develop` | Branch de integração contínua das funcionalidades | `integra feature/* na develop` |
| `feature/fase1-pipeline-ml` | Treinamento offline, pré-processamento e serialização `.joblib` | `implementa pipeline de machine learning e serializa modelo` |
| `feature/fase4-schemas-pydantic` | Schemas de entrada e saída com restrições `Field` e `Enum` | `implementa schemas pydantic de entrada e saida com validacao field` |
| `feature/fase2-3-service-lifespan` | Camada de serviço isolada e carregamento único em memória RAM | `implementa camada de servico para inferencia em memoria ram` |
| `feature/fase5-endpoints-dashboard` | Rotas `/health` e `/predict`, `lifespan`, Dashboard Streamlit, Colab e Testes | `implementa endpoints fastapi com lifespan, dashboard streamlit e documentacao` |

---

## 8. Links das Evidências

- 🎥 **Vídeo de Apresentação e Demonstração (Google Drive — Modo Leitor Público)**:  
  👉 `https://drive.google.com/file/d/1UywA-aP4YJ67--UV6b15qPkwhVLmvO4-/view?usp=sharing`
- 💻 **Repositório Público no GitHub.io**:  
  👉 `https://jorgerodrigues2026.github.io/credito-dashboard/]`
- 📄 **Caderno Técnico Completo em PDF (Códigos Python + Explicações)**:  
  👉 [`resumo.PDF`](./resumo.PDF)
- 📓 **Jupyter Notebook Interativo (Google Colab)**:  
  👉 [`projeto_predicao_credito_colab.ipynb`](./projeto_predicao_credito_colab.ipynb)
