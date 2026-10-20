"""
====================================================================================
FASE 2 & FASE 5: IMPLEMENTAÇÃO DE ENDPOINTS E TRATAMENTO DE ERROS (FASTAPI & HTTP)
Evolução do arquivo base `predict_routes.py`
====================================================================================
Implementa:
1. `GET /health`: Monitoramento operacional verificando se o serviço está ativo e se
   o modelo de Machine Learning reside corretamente em memória RAM (HTTP 200 OK).
2. `POST /predict` e `POST /v1/predict`: Endpoints de inferência sob demanda que
   recebem o schema validado `DadosClienteInput`, delegam o cálculo para a camada
   `services/` (sem ler o disco a cada requisição!) e retornam `PredicaoCreditoOutput`.
3. Tratamento semântico de exceções HTTP:
   - 200 OK: Inferência ou healthcheck concluído com sucesso.
   - 400 Bad Request: Inconsistência nas regras de negócio financeiras.
   - 422 Unprocessable Entity: Violação de tipos ou limites Pydantic (`Field`).
   - 500 Internal Server Error: Erro inesperado tratado com segurança sem derrubar a API.
====================================================================================
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas.cliente import DadosClienteInput, HealthResponse, PredicaoCreditoOutput
from app.services.credito_service import (
    ModeloNaoCarregadoException,
    RegraNegocioException,
    model_service,
)

# Roteador dedicado para Monitoramento de Saúde
health_router = APIRouter(tags=["Monitoramento e Saúde"])

# Roteador dedicado para Inferência Preditiva (Evolução de `predict_route`)
predict_router = APIRouter(tags=["Predição de Crédito"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Verifica a integridade operacional da API e do modelo em memória RAM",
)
def verificar_saude() -> HealthResponse:
    """
    Endpoint de monitoramento (`GET /health`).
    Responde com HTTP 200 OK informando se a API está ativa e se o modelo preditivo
    foi carregado na memória RAM durante o ciclo de vida (`lifespan`).
    """
    status_atual = model_service.obter_status_saude()
    return HealthResponse(**status_atual)


def _executar_fluxo_predicao(cliente: DadosClienteInput) -> PredicaoCreditoOutput:
    """Função auxiliar compartilhada entre `/predict` e `/v1/predict`."""
    try:
        return model_service.realizar_predicao(cliente)
    except RegraNegocioException as erro_negocio:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(erro_negocio),
        ) from erro_negocio
    except ModeloNaoCarregadoException as erro_modelo:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Serviço de inferência indisponível em memória: {str(erro_modelo)}",
        ) from erro_modelo
    except Exception as erro_interno:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno no processamento da predição: {str(erro_interno)}",
        ) from erro_interno


@predict_router.post(
    "/predict",
    response_model=PredicaoCreditoOutput,
    status_code=status.HTTP_200_OK,
    summary="Avalia o risco de crédito e retorna a decisão preditiva (Rota Padrão)",
    responses={
        200: {"description": "Predição realizada com sucesso pelo modelo em RAM."},
        400: {"description": "Violação de regra de negócio financeira (inconsistência cadastral)."},
        422: {"description": "Erro de validação estrita de tipos ou limites de entrada (Pydantic)."},
        500: {"description": "Falha interna controlada no serviço de inferência."},
    },
)
def prever_aprovacao(cliente: DadosClienteInput) -> PredicaoCreditoOutput:
    """
    Recebe as características validadas do cliente e devolve se o crédito foi
    **Aprovado** ou **Reprovado**, juntamente com a probabilidade estimada e faixa de risco.
    """
    return _executar_fluxo_predicao(cliente)


@predict_router.post(
    "/v1/predict",
    response_model=PredicaoCreditoOutput,
    status_code=status.HTTP_200_OK,
    summary="Avalia o risco de crédito e retorna a decisão preditiva (Rota Versionada v1)",
    responses={
        200: {"description": "Predição realizada com sucesso pelo modelo em RAM."},
        400: {"description": "Violação de regra de negócio financeira (inconsistência cadastral)."},
        422: {"description": "Erro de validação estrita de tipos ou limites de entrada (Pydantic)."},
        500: {"description": "Falha interna controlada no serviço de inferência."},
    },
)
def prever_aprovacao_v1(cliente: DadosClienteInput) -> PredicaoCreditoOutput:
    """
    Endpoint versionado (`/v1/predict`) com o mesmo contrato estrito de entrada e saída.
    """
    return _executar_fluxo_predicao(cliente)
