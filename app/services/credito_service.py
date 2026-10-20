"""
====================================================================================
FASE 2 & FASE 3: CAMADA DE SERVIÇO E GERENCIAMENTO EM MEMÓRIA RAM
====================================================================================
Desacopla totalmente a lógica matemática e de Machine Learning das rotas HTTP.
Respeita estritamente a Proibição Técnica da Fase 3:
- O arquivo `models/modelo_emprestimo.joblib` é carregado UMA ÚNICA VEZ durante
  o evento de startup do `@asynccontextmanager` (`lifespan` no `main.py`).
- O método `realizar_predicao` apenas consome o objeto já residente em memória RAM,
  sem qualquer operação de leitura de disco (I/O) por requisição.
====================================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import pandas as pd

from app.schemas.cliente import DadosClienteInput, PredicaoCreditoOutput

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "modelo_emprestimo.joblib"


class RegraNegocioException(ValueError):
    """Exceção lançada quando os dados violam uma invariante semântica de negócio (HTTP 400)."""


class ModeloNaoCarregadoException(RuntimeError):
    """Exceção lançada caso o modelo não esteja disponível em memória RAM (HTTP 500/503)."""


class CreditoModelService:
    """
    Serviço Singleton responsável por manter o Pipeline Scikit-Learn em memória RAM
    e executar predições de baixa latência.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path: Path = model_path
        self._pipeline: Optional[Any] = None
        self._versao_modelo: str = "desconhecida"
        self._algoritmo: str = "desconhecido"
        self._metricas_teste: Dict[str, Any] = {}

    @property
    def esta_carregado(self) -> bool:
        """Retorna True se o pipeline preditivo está instanciado na memória RAM."""
        return self._pipeline is not None

    def carregar_modelo_em_ram(self) -> None:
        """
        Carrega o artefato `.joblib` do disco para a memória RAM.
        Chamado exclusivamente no ciclo de vida (`lifespan`) na inicialização do servidor.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Artefato do modelo não encontrado em '{self.model_path}'. "
                "Execute `python train_pipeline.py` antes de iniciar a API."
            )

        artefato = joblib.load(self.model_path)

        # Suporta tanto o dicionário empacotado com metadados quanto um modelo puro do Scikit-Learn
        if isinstance(artefato, dict) and "pipeline" in artefato:
            self._pipeline = artefato["pipeline"]
            self._versao_modelo = artefato.get("versao_modelo", "1.0.0")
            self._algoritmo = artefato.get("algoritmo", type(self._pipeline).__name__)
            self._metricas_teste = artefato.get("metricas_teste", {})
        else:
            self._pipeline = artefato
            self._versao_modelo = "1.0.0-legado"
            self._algoritmo = type(artefato).__name__
            self._metricas_teste = {}

    def descarregar_modelo_da_ram(self) -> None:
        """Libera a referência do modelo em memória durante o encerramento da aplicação."""
        self._pipeline = None

    def obter_status_saude(self) -> Dict[str, Any]:
        """Retorna os metadados operacionais para o endpoint GET /health."""
        return {
            "status_servico": "online" if self.esta_carregado else "degradado",
            "modelo_carregado_em_ram": self.esta_carregado,
            "algoritmo": self._algoritmo if self.esta_carregado else None,
            "versao_modelo": self._versao_modelo if self.esta_carregado else None,
            "metricas_referencia_teste": self._metricas_teste if self.esta_carregado else None,
        }

    def realizar_predicao(self, cliente: DadosClienteInput) -> PredicaoCreditoOutput:
        """
        Executa a inferência preditiva em memória RAM a partir do schema validado.
        Também aplica validações semânticas cruzadas de domínio financeiro (HTTP 400).
        """
        if not self.esta_carregado or self._pipeline is None:
            raise ModeloNaoCarregadoException(
                "O modelo preditivo não está carregado na memória RAM."
            )

        # Validação semântica de negócio (gera HTTP 400 Bad Request na camada de rota):
        # 1. Tempo de emprego não pode exceder a idade laboral possível (idade - 14 anos)
        idade_laboral_maxima = max(0, cliente.idade - 14)
        if cliente.tempo_emprego_anos > idade_laboral_maxima:
            raise RegraNegocioException(
                f"Inconsistência cadastral: tempo_emprego_anos ({cliente.tempo_emprego_anos}) "
                f"não pode ser maior que a idade laboral possível ({idade_laboral_maxima} anos "
                f"para um cliente de {cliente.idade} anos)."
            )

        # 2. Valor do empréstimo não pode ultrapassar 20 vezes a renda anual declarada
        renda_anual = cliente.renda_mensal * 12.0
        indice_comprometimento = cliente.valor_emprestimo / renda_anual
        if indice_comprometimento > 20.0:
            raise RegraNegocioException(
                f"Solicitação desenquadrada da política de crédito: o valor solicitado "
                f"(R$ {cliente.valor_emprestimo:.2f}) excede em {indice_comprometimento:.1f}x "
                f"a renda anual declarada (R$ {renda_anual:.2f})."
            )

        # Montagem do DataFrame 2D com os nomes exatos das colunas esperadas pelo ColumnTransformer
        dados_entrada = pd.DataFrame(
            [
                {
                    "idade": float(cliente.idade),
                    "pontuacao_credito": float(cliente.pontuacao_credito),
                    "renda_mensal": float(cliente.renda_mensal),
                    "valor_emprestimo": float(cliente.valor_emprestimo),
                    "tempo_emprego_anos": float(cliente.tempo_emprego_anos),
                    "inadimplencia_previa": int(cliente.inadimplencia_previa),
                    "tipo_moradia": cliente.tipo_moradia.value,
                    "finalidade_emprestimo": cliente.finalidade_emprestimo.value,
                }
            ]
        )

        # Inferência em memória RAM (sem leitura de disco!)
        resultado_classe = int(self._pipeline.predict(dados_entrada)[0])

        if hasattr(self._pipeline, "predict_proba"):
            probabilidades = self._pipeline.predict_proba(dados_entrada)[0]
            prob_aprovacao = float(probabilidades[1])
            prob_risco = float(probabilidades[0])
        else:
            prob_aprovacao = 1.0 if resultado_classe == 1 else 0.0
            prob_risco = 1.0 - prob_aprovacao

        status_decisao = "Aprovado" if resultado_classe == 1 else "Reprovado"

        if prob_aprovacao >= 0.75:
            faixa_risco = "Baixo Risco"
        elif prob_aprovacao >= 0.45:
            faixa_risco = "Risco Moderado"
        else:
            faixa_risco = "Alto Risco"

        return PredicaoCreditoOutput(
            nome_cliente=cliente.nome_cliente,
            idade_informada=cliente.idade,
            pontuacao_informada=cliente.pontuacao_credito,
            decisao=status_decisao,
            codigo_decisao=resultado_classe,
            probabilidade_aprovacao=round(prob_aprovacao, 4),
            probabilidade_risco_inadimplencia=round(prob_risco, 4),
            faixa_risco=faixa_risco,
            indice_comprometimento_anual=round(indice_comprometimento, 4),
            versao_modelo=self._versao_modelo,
        )


# Instância única compartilhada com o Lifespan (main.py) e as Rotas (routers/)
model_service = CreditoModelService()
