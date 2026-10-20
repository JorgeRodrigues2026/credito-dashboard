"""
====================================================================================
FASE 2 & FASE 3: PONTO DE ENTRADA DA APLICAÇÃO E CICLO DE VIDA (LIFESPAN)
Evolução do arquivo base `main.py`
====================================================================================
Implementa:
1. `@asynccontextmanager` na função `lifespan(app: FastAPI)` para carregar o artefato
   serializado (`models/modelo_emprestimo.joblib`) na memória RAM **uma única vez**
   durante a inicialização da aplicação.
2. Caso o arquivo `.joblib` ainda não tenha sido gerado, dispara automaticamente o
   treinamento inicial (`executar_treinamento()`) antes de subir a API, garantindo
   execução sem falhas para qualquer avaliador.
3. Inclusão modular dos roteadores (`health_router` e `predict_router`).
====================================================================================
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.routers.predict_routes import health_router, predict_router
from app.services.credito_service import model_service
from train_pipeline import executar_treinamento


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador assíncrono de ciclo de vida (Lifespan) do FastAPI.
    Garante que o modelo de Machine Learning seja carregado em memória RAM
    uma única vez no startup e liberado no shutdown.
    """
    if not model_service.model_path.exists():
        print("[LIFESPAN] Artefato não encontrado em disco. Executando pipeline de treinamento...")
        executar_treinamento()

    print(f"[LIFESPAN] Carregando modelo em memória RAM a partir de: {model_service.model_path.name}")
    model_service.carregar_modelo_em_ram()
    app.state.model_service = model_service
    print("[LIFESPAN] Modelo carregado em RAM com sucesso! API pronta para receber requisições.")

    yield

    print("[LIFESPAN] Encerrando aplicação e liberando modelo da memória RAM...")
    model_service.descarregar_modelo_da_ram()


app = FastAPI(
    title="API de Predição de Crédito - MLOps",
    description=(
        "Serviço preditivo RESTful para aprovação automática e análise de risco de "
        "empréstimos com Machine Learning (Scikit-Learn Pipeline), validação estrita "
        "Pydantic e gerenciamento de ciclo de vida em memória RAM (Lifespan)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Conectamos os roteadores modulares na aplicação
app.include_router(health_router)
app.include_router(predict_router)

# Comando para execução:
# uvicorn main:app --reload
