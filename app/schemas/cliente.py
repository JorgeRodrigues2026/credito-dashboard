"""
====================================================================================
FASE 4: MODELAGEM E VALIDAÇÃO ESTRITA DE DADOS (PYDANTIC)
Evolução do arquivo base `cliente.py` (`DadosCliente`)
====================================================================================
Implementa contratos estritos de Entrada (`DadosClienteInput`) e Saída
(`PredicaoCreditoOutput` e `HealthResponse`) utilizando `BaseModel`, `Field` e `Enum`.
Qualquer violação de tipo ou limite de negócio (ex: idade < 18, score > 850, texto em
campo numérico) é interceptada automaticamente pelo Pydantic retornando HTTP 422
(Unprocessable Entity) antes de chegar ao modelo de Machine Learning.
====================================================================================
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TipoMoradiaEnum(str, Enum):
    """Categorias válidas de situação habitacional aceitas pelo pipeline."""

    PROPRIA = "PROPRIA"
    ALUGADA = "ALUGADA"
    FINANCIADA = "FINANCIADA"
    OUTROS = "OUTROS"


class FinalidadeEmprestimoEnum(str, Enum):
    """Finalidades válidas de solicitação de crédito aceitas pelo pipeline."""

    EDUCACAO = "EDUCACAO"
    SAUDE = "SAUDE"
    NEGOCIOS = "NEGOCIOS"
    PESSOAL = "PESSOAL"
    QUITACAO_DIVIDAS = "QUITACAO_DIVIDAS"
    IMOVEL = "IMOVEL"


class DadosClienteInput(BaseModel):
    """
    Schema de Entrada para avaliação preditiva de crédito (Evolução de `DadosCliente`).
    Possui valores padrão nos campos expandidos para manter retrocompatibilidade caso
    um cliente envie apenas `idade` e `pontuacao_credito`, mas valida rigorosamente
    todos os atributos enviados.
    """

    nome_cliente: Optional[str] = Field(
        default="Cliente Padrão",
        min_length=2,
        max_length=80,
        description="Nome ou identificador do solicitante (2 a 80 caracteres).",
        examples=["Maria Oliveira"],
    )
    idade: int = Field(
        ...,
        ge=18,
        le=100,
        description="Idade do cliente em anos completos (permitido entre 18 e 100 anos).",
        examples=[34],
    )
    pontuacao_credito: int = Field(
        ...,
        ge=300,
        le=850,
        description="Score de crédito de mercado (faixa FICO/Serasa entre 300 e 850).",
        examples=[710],
    )
    renda_mensal: float = Field(
        default=5000.0,
        gt=0.0,
        le=1000000.0,
        description="Renda mensal bruta comprovada em Reais (R$ > 0).",
        examples=[6500.00],
    )
    valor_emprestimo: float = Field(
        default=12000.0,
        ge=500.0,
        le=500000.0,
        description="Valor total solicitado para empréstimo em Reais (entre R$ 500 e R$ 500.000).",
        examples=[15000.00],
    )
    tempo_emprego_anos: float = Field(
        default=3.0,
        ge=0.0,
        le=60.0,
        description="Tempo de estabilidade no emprego/atividade atual em anos (0 a 60).",
        examples=[4.5],
    )
    inadimplencia_previa: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Histórico de inadimplência prévia nos últimos 24 meses (0 = Não, 1 = Sim).",
        examples=[0],
    )
    tipo_moradia: TipoMoradiaEnum = Field(
        default=TipoMoradiaEnum.PROPRIA,
        description="Situação de moradia atual do solicitante.",
        examples=[TipoMoradiaEnum.PROPRIA],
    )
    finalidade_emprestimo: FinalidadeEmprestimoEnum = Field(
        default=FinalidadeEmprestimoEnum.PESSOAL,
        description="Finalidade declarada para o uso do crédito.",
        examples=[FinalidadeEmprestimoEnum.NEGOCIOS],
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "nome_cliente": "Carlos Eduardo Santos",
                "idade": 35,
                "pontuacao_credito": 720,
                "renda_mensal": 7500.0,
                "valor_emprestimo": 18000.0,
                "tempo_emprego_anos": 5.0,
                "inadimplencia_previa": 0,
                "tipo_moradia": "PROPRIA",
                "finalidade_emprestimo": "NEGOCIOS",
            }
        },
    )


# Alias mantendo exatamente o nome da classe do arquivo base `cliente.py`
DadosCliente = DadosClienteInput


class PredicaoCreditoOutput(BaseModel):
    """
    Schema Estrito de Saída da Predição.
    Preserva os campos do código base (`idade_informada`, `pontuacao_informada`,
    `decisao`, `codigo_decisao`) e adiciona probabilidades e faixa de risco.
    """

    nome_cliente: Optional[str] = Field(
        default=None,
        description="Identificador do solicitante avaliado.",
    )
    idade_informada: int = Field(
        ...,
        ge=18,
        le=100,
        description="Idade recebida na requisição.",
    )
    pontuacao_informada: int = Field(
        ...,
        ge=300,
        le=850,
        description="Score de crédito recebido na requisição.",
    )
    decisao: str = Field(
        ...,
        description="Status de negócio legível: 'Aprovado' ou 'Reprovado'.",
        examples=["Aprovado"],
    )
    codigo_decisao: int = Field(
        ...,
        ge=0,
        le=1,
        description="Código binário da classe predita (1 = Aprovado, 0 = Reprovado).",
        examples=[1],
    )
    probabilidade_aprovacao: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada pelo modelo para aprovação (0.0 a 1.0).",
        examples=[0.8742],
    )
    probabilidade_risco_inadimplencia: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade complementar de risco/reprovação (0.0 a 1.0).",
        examples=[0.1258],
    )
    faixa_risco: str = Field(
        ...,
        description="Classificação qualitativa de risco ('Baixo Risco', 'Risco Moderado', 'Alto Risco').",
        examples=["Baixo Risco"],
    )
    indice_comprometimento_anual: float = Field(
        ...,
        ge=0.0,
        description="Razão entre o valor solicitado e a renda anual do cliente.",
        examples=[0.2],
    )
    versao_modelo: str = Field(
        ...,
        description="Versão do artefato de Machine Learning em memória que gerou a predição.",
        examples=["1.0.0"],
    )


class HealthResponse(BaseModel):
    """Schema de Saída do endpoint de monitoramento GET /health."""

    status_servico: str = Field(
        ...,
        description="Estado operacional da API HTTP ('online' ou 'degradado').",
        examples=["online"],
    )
    modelo_carregado_em_ram: bool = Field(
        ...,
        description="Indica se o pipeline preditivo reside ativamente na memória RAM.",
        examples=[True],
    )
    algoritmo: Optional[str] = Field(
        default=None,
        description="Identificação do pipeline carregado.",
        examples=["RandomForestClassifier (Scikit-Learn Pipeline)"],
    )
    versao_modelo: Optional[str] = Field(
        default=None,
        description="Versão do modelo carregado na inicialização (Lifespan).",
        examples=["1.0.0"],
    )
    metricas_referencia_teste: Optional[dict] = Field(
        default=None,
        description="Métricas de validação em conjunto de teste registradas no artefato.",
    )
