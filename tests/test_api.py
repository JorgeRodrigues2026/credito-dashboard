"""
Suíte automatizada de testes (Pytest + FastAPI TestClient).
Valida:
1. Carregamento do modelo em memória RAM via Lifespan (`GET /health` -> 200 OK).
2. Predição para cliente de baixo risco (`POST /predict` -> 200 OK, Aprovado).
3. Predição para cliente de alto risco (`POST /v1/predict` -> 200 OK, Reprovado).
4. Retrocompatibilidade com o payload mínimo dos arquivos base (`idade` e `pontuacao_credito`).
5. Interceptação automática de tipos e limites inválidos pelo Pydantic (422 Unprocessable Entity).
6. Tratamento semântico de inconsistência de regra de negócio financeira (400 Bad Request).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from main import app


def test_health_endpoint_modelo_em_ram():
    """Verifica se o Lifespan carregou o modelo em RAM e se /health responde 200 OK."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        dados = response.json()
        assert dados["status_servico"] == "online"
        assert dados["modelo_carregado_em_ram"] is True
        assert dados["versao_modelo"] == "1.0.0"


def test_predict_cliente_aprovado_200():
    """Verifica inferência de aprovação com payload completo válido."""
    payload = {
        "nome_cliente": "Ana Beatriz Souza",
        "idade": 38,
        "pontuacao_credito": 780,
        "renda_mensal": 12000.0,
        "valor_emprestimo": 15000.0,
        "tempo_emprego_anos": 8.0,
        "inadimplencia_previa": 0,
        "tipo_moradia": "PROPRIA",
        "finalidade_emprestimo": "NEGOCIOS",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        dados = response.json()
        assert dados["decisao"] == "Aprovado"
        assert dados["codigo_decisao"] == 1
        assert 0.0 <= dados["probabilidade_aprovacao"] <= 1.0


def test_predict_v1_cliente_reprovado_200():
    """Verifica inferência de reprovação na rota /v1/predict."""
    payload = {
        "nome_cliente": "Roberto Silva",
        "idade": 21,
        "pontuacao_credito": 360,
        "renda_mensal": 1800.0,
        "valor_emprestimo": 45000.0,
        "tempo_emprego_anos": 0.5,
        "inadimplencia_previa": 1,
        "tipo_moradia": "ALUGADA",
        "finalidade_emprestimo": "QUITACAO_DIVIDAS",
    }
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        dados = response.json()
        assert dados["decisao"] == "Reprovado"
        assert dados["codigo_decisao"] == 0


def test_predict_payload_minimo_retrocompativel_200():
    """Garante compatibilidade com o contrato original (`idade` e `pontuacao_credito`)."""
    payload_minimo = {
        "idade": 42,
        "pontuacao_credito": 740,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload_minimo)
        assert response.status_code == 200
        dados = response.json()
        assert dados["idade_informada"] == 42
        assert dados["pontuacao_informada"] == 740
        assert dados["decisao"] in ["Aprovado", "Reprovado"]


def test_predict_erro_validacao_pydantic_422():
    """Garante que dados fora dos limites (Field) ou tipos errados gerem 422 Unprocessable Entity."""
    payload_invalido = {
        "idade": 15,  # Violação: ge=18
        "pontuacao_credito": 950,  # Violação: le=850
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload_invalido)
        assert response.status_code == 422


def test_predict_erro_regra_negocio_400():
    """Garante que inconsistência lógica de negócio retorne 400 Bad Request."""
    payload_inconsistente = {
        "idade": 20,
        "pontuacao_credito": 700,
        "renda_mensal": 4000.0,
        "valor_emprestimo": 10000.0,
        "tempo_emprego_anos": 15.0,  # Impossível ter 15 anos de emprego aos 20 anos de idade
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload_inconsistente)
        assert response.status_code == 400
        assert "Inconsistência cadastral" in response.json()["detail"]
