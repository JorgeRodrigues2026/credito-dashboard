from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Caminhos relativos (Item 5.1: Proibido caminhos absolutos da maquina)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "credito_dataset.csv"
MODEL_PATH = MODELS_DIR / "modelo_emprestimo.joblib"
METRICS_PATH = MODELS_DIR / "relatorio_metricas.json"


def gerar_ou_carregar_dados(
    caminho_csv: Path,
    n_amostras: int = 3000,
) -> pd.DataFrame:
    """Carrega o CSV local ou gera a base realista de Risco de Credito."""
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    if caminho_csv.exists():
        return pd.read_csv(caminho_csv)

    rng = np.random.default_rng(seed=42)
    idade = rng.integers(18, 75, size=n_amostras).astype(float)
    pontuacao_credito = np.clip(
        rng.normal(620, 110, size=n_amostras), 300, 850
    ).round().astype(float)
    renda_mensal = np.clip(
        rng.lognormal(mean=8.3, sigma=0.55, size=n_amostras), 1200, 50000
    ).round(2)
    valor_emprestimo = np.clip(
        rng.lognormal(mean=9.4, sigma=0.7, size=n_amostras), 1000, 150000
    ).round(2)
    tempo_emprego_anos = np.clip(
        rng.exponential(scale=5.0, size=n_amostras), 0, 40
    ).round(1)
    inadimplencia_previa = rng.choice(
        [0, 1], size=n_amostras, p=[0.82, 0.18]
    )
    tipo_moradia = rng.choice(
        ["PROPRIA", "ALUGADA", "FINANCIADA", "OUTROS"],
        size=n_amostras,
        p=[0.40, 0.35, 0.20, 0.05],
    )
    finalidade_emprestimo = rng.choice(
        ["EDUCACAO", "SAUDE", "NEGOCIOS", "PESSOAL", "QUITACAO_DIVIDAS", "IMOVEL"],
        size=n_amostras,
        p=[0.18, 0.12, 0.20, 0.22, 0.18, 0.10],
    )
    comprometimento = valor_emprestimo / (renda_mensal * 12.0)
    score_latente = (
        0.018 * (pontuacao_credito - 600)
        + 0.015 * (idade - 30)
        + 0.08 * tempo_emprego_anos
        - 3.8 * comprometimento
        - 2.1 * inadimplencia_previa
        + np.where(tipo_moradia == "PROPRIA", 0.55, 0.0)
        + np.where(tipo_moradia == "ALUGADA", -0.35, 0.0)
        + np.where(finalidade_emprestimo == "QUITACAO_DIVIDAS", -0.45, 0.2)
        + rng.normal(0, 0.65, size=n_amostras)
    )
    prob_aprovacao = 1 / (1 + np.exp(-score_latente))
    aprovado = (prob_aprovacao >= 0.5).astype(int)
    df = pd.DataFrame({
        "idade": idade,
        "pontuacao_credito": pontuacao_credito,
        "renda_mensal": renda_mensal,
        "valor_emprestimo": valor_emprestimo,
        "tempo_emprego_anos": tempo_emprego_anos,
        "inadimplencia_previa": inadimplencia_previa,
        "tipo_moradia": tipo_moradia,
        "finalidade_emprestimo": finalidade_emprestimo,
        "aprovado": aprovado,
    })
    for col in ["renda_mensal", "tempo_emprego_anos", "tipo_moradia"]:
        mask_nulos = rng.random(n_amostras) < 0.03
        df.loc[mask_nulos, col] = np.nan
    df.to_csv(caminho_csv, index=False)
    return df


def construir_pipeline() -> Pipeline:
    """Constroi o Pipeline de Pre-processamento + Classificacao."""
    colunas_numericas = [
        "idade", "pontuacao_credito", "renda_mensal",
        "valor_emprestimo", "tempo_emprego_anos", "inadimplencia_previa",
    ]
    colunas_categoricas = ["tipo_moradia", "finalidade_emprestimo"]
    transformador_numerico = Pipeline(steps=[
        ("imputador_mediana", SimpleImputer(strategy="median")),
        ("escalonador_padrao", StandardScaler()),
    ])
    transformador_categorico = Pipeline(steps=[
        ("imputador_moda", SimpleImputer(strategy="most_frequent")),
        ("codificador_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, colunas_numericas),
        ("cat", transformador_categorico, colunas_categoricas),
    ])
    return Pipeline(steps=[
        ("preprocessamento", preprocessador),
        ("classificador", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def avaliar_particao(modelo: Pipeline, x_p: pd.DataFrame, y_p: pd.Series, nome: str) -> dict:
    y_pred = modelo.predict(x_p)
    y_prob = modelo.predict_proba(x_p)[:, 1]
    return {
        "particao": nome,
        "amostras": int(len(y_p)),
        "acuracia": round(float(accuracy_score(y_p, y_pred)), 4),
        "precisao": round(float(precision_score(y_p, y_pred)), 4),
        "recall": round(float(recall_score(y_p, y_pred)), 4),
        "f1_score": round(float(f1_score(y_p, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_p, y_prob)), 4),
        "matriz_confusao": confusion_matrix(y_p, y_pred).tolist(),
    }


def executar_treinamento() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = gerar_ou_carregar_dados(DATASET_PATH)
    X = df.drop(columns=["aprovado"])
    y = df["aprovado"]
    X_treino, X_temp, y_treino, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_teste, y_val, y_teste = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    pipeline = construir_pipeline()
    pipeline.fit(X_treino, y_treino)
    m_treino = avaliar_particao(pipeline, X_treino, y_treino, "treino")
    m_val = avaliar_particao(pipeline, X_val, y_val, "validacao")
    m_teste = avaliar_particao(pipeline, X_teste, y_teste, "teste")
    print("=== RELATORIO DE CLASSIFICACAO (CONJUNTO DE TESTE - 15%) ===")
    print(classification_report(y_teste, pipeline.predict(X_teste), target_names=["Reprovado (0)", "Aprovado (1)"]))
    artefato = {
        "pipeline": pipeline,
        "versao_modelo": "1.0.0",
        "algoritmo": "RandomForestClassifier (Scikit-Learn Pipeline)",
        "features_esperadas": list(X.columns),
        "metricas_teste": m_teste,
    }
    joblib.dump(artefato, MODEL_PATH)
    relatorio = {"versao_modelo": "1.0.0", "metricas": {"treino": m_treino, "validacao": m_val, "teste": m_teste}}
    METRICS_PATH.write_text(json.dumps(relatorio, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[SUCESSO] Modelo salvo em: {MODEL_PATH.relative_to(BASE_DIR)}")
    print(f"[SUCESSO] Metricas salvas em: {METRICS_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    executar_treinamento()
