"""
====================================================================================
DASHBOARD EXECUTIVO & SIMULADOR INTERATIVO (STREAMLIT + FASTAPI REST)
Projeto: Sistema Preditivo de Análise de Risco e Concessão de Crédito (MLOps)
====================================================================================
Funcionalidades do Dashboard:
1. Monitoramento em Tempo Real da API (`GET /health`):
   - Exibe status do servidor FastAPI, presença do modelo na memória RAM (`lifespan`),
     versão do artefato e métricas oficiais do modelo em cards executivos.
2. Simulador Preditivo Interativo (`POST /predict` e `POST /v1/predict`):
   - Formulário rico com presets de perfis (Baixo Risco, Moderado, Alto Risco),
     gráfico velocímetro (Gauge) de probabilidade de aprovação, indicadores financeiros
     e inspetor de Payload JSON (Request/Response HTTP).
3. Laboratório de Validação & Governança HTTP (Demonstração para o Vídeo Avaliativo!):
   - Botões de 1 clique que disparam requisições reais para a API FastAPI demonstrando
     ao vivo os retornos `200 OK`, `422 Unprocessable Entity` (Pydantic Field) e
     `400 Bad Request` (Regra de Negócio Financeira).
4. Analytics & Performance do Modelo (Fase 1):
   - Gráficos interativos Plotly com Matriz de Confusão, Comparativo Treino/Validação/Teste
     e exploração da base `data/credito_dataset.csv`.
====================================================================================
"""

from pathlib import Path
import json
import time
from typing import Any, Dict, Tuple
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "credito_dataset.csv"
METRICS_PATH = BASE_DIR / "models" / "relatorio_metricas.json"

st.set_page_config(
    page_title="CreditAI MLOps | Dashboard & Simulador FastAPI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS profissional para Cards e Badges HTTP
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: #f8fafc;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.98rem;
    }
    .status-badge-ok {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #86efac;
    }
    .status-badge-err {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #fca5a5;
    }
    .decision-card-approved {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 6px solid #10b981;
        padding: 1.25rem;
        border-radius: 10px;
        color: #065f46;
    }
    .decision-card-rejected {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
        padding: 1.25rem;
        border-radius: 10px;
        color: #7f1d1d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def consultar_health_api(base_url: str) -> Tuple[bool, int, Dict[str, Any], float]:
    """Chama o endpoint GET /health da API FastAPI e mede a latência em ms."""
    inicio = time.perf_counter()
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(f"{base_url.rstrip('/')}/health")
            latencia_ms = (time.perf_counter() - inicio) * 1000.0
            return resp.status_code == 200, resp.status_code, resp.json(), latencia_ms
    except Exception as exc:
        latencia_ms = (time.perf_counter() - inicio) * 1000.0
        return (
            False,
            0,
            {"erro_conexao": str(exc), "dica": "Inicie a API com: uvicorn main:app --reload"},
            latencia_ms,
        )


def chamar_predict_api(
    base_url: str, endpoint: str, payload: Dict[str, Any]
) -> Tuple[int, Dict[str, Any], float]:
    """Envia requisição POST /predict ou /v1/predict para a API FastAPI."""
    inicio = time.perf_counter()
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)
            latencia_ms = (time.perf_counter() - inicio) * 1000.0
            return resp.status_code, resp.json(), latencia_ms
    except Exception as exc:
        latencia_ms = (time.perf_counter() - inicio) * 1000.0
        return (
            503,
            {
                "detail": f"Não foi possível conectar ao servidor FastAPI em {url}: {str(exc)}. "
                "Certifique-se de que o comando `uvicorn main:app --reload` está ativo."
            },
            latencia_ms,
        )


def criar_grafico_velocimetro(prob_aprovacao: float, decisao: str) -> go.Figure:
    """Cria um velocímetro (Gauge) Plotly para a probabilidade de aprovação de crédito."""
    cor_barra = "#10b981" if decisao == "Aprovado" else "#ef4444"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_aprovacao * 100.0,
            number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
            title={"text": "Probabilidade de Aprovação (Modelo em RAM)", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": cor_barra, "thickness": 0.35},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 45], "color": "#fee2e2"},
                    {"range": [45, 75], "color": "#fef3c7"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 4},
                    "thickness": 0.8,
                    "value": 50.0,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=25, r=25, t=45, b=15))
    return fig


# ==============================================================================
# BARRA LATERAL (CONFIGURAÇÃO DA API FASTAPI E MONITORAMENTO DE SAÚDE)
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Conexão REST FastAPI")
    api_base_url = st.text_input(
        "URL Base da API FastAPI",
        value="http://127.0.0.1:8000",
        help="Endereço onde o servidor Uvicorn/FastAPI está rodando.",
    )
    rota_escolhida = st.selectbox(
        "Endpoint de Inferência",
        options=["/predict", "/v1/predict"],
        index=0,
        help="Selecione a rota padrão ou a rota versionada v1 da API.",
    )

    btn_refresh = st.button("🔄 Verificar Saúde (/health)", use_container_width=True)

    online, status_health, dados_health, lat_health = consultar_health_api(api_base_url)

    st.markdown("---")
    st.markdown("### 🩺 Status Operacional (`GET /health`)")
    if online:
        st.markdown(
            f'<span class="status-badge-ok">● ONLINE (HTTP {status_health}) • {lat_health:.1f} ms</span>',
            unsafe_allow_html=True,
        )
        em_ram = dados_health.get("modelo_carregado_em_ram", False)
        st.metric(
            label="Modelo Residente em Memória RAM",
            value="ATIVO (Lifespan)" if em_ram else "INATIVO",
            delta=f"Versão {dados_health.get('versao_modelo', '1.0.0')}",
        )
        st.caption(f"**Algoritmo:** {dados_health.get('algoritmo', 'N/A')}")
    else:
        st.markdown(
            '<span class="status-badge-err">● OFFLINE (API Indisponível)</span>',
            unsafe_allow_html=True,
        )
        st.warning(
            "Inicie o servidor FastAPI em outro terminal:\n"
            "```bash\nuvicorn main:app --reload\n```"
        )

    st.markdown("---")
    st.markdown(
        "🔗 **Atalhos da Documentação FastAPI:**\n"
        f"- [Swagger UI (`/docs`)]({api_base_url}/docs)\n"
        f"- [ReDoc (`/redoc`)]({api_base_url}/redoc)\n"
        f"- [Health JSON (`/health`)]({api_base_url}/health)"
    )

# ==============================================================================
# CABEÇALHO PRINCIPAL E HISTÓRICO DE SESSÃO
# ==============================================================================
if "historico_requisicoes" not in st.session_state:
    st.session_state["historico_requisicoes"] = []

st.markdown(
    """
    <div class="main-header">
        <h1>🏦 CreditAI MLOps — Plataforma de Análise Preditiva de Crédito</h1>
        <p>Interface Executiva em Streamlit integrada via HTTP REST ao Microsserviço FastAPI, Validação Estrita Pydantic e Pipeline Scikit-Learn em Memória RAM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Barra de KPIs no Topo
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
metricas_ref = dados_health.get("metricas_referencia_teste", {}) if online else {}
with kpi1:
    st.metric(
        "Status do Microsserviço",
        "200 OK (Online)" if online else "Offline",
        f"{lat_health:.1f} ms (/health)" if online else "Sem conexão",
    )
with kpi2:
    roc_val = metricas_ref.get("roc_auc", 0.936)
    st.metric("ROC-AUC (Conjunto Teste)", f"{roc_val * 100:.2f}%", "Meta > 85%")
with kpi3:
    f1_val = metricas_ref.get("f1_score", 0.871)
    st.metric("F1-Score Balanceado", f"{f1_val * 100:.2f}%", "Classes Ponderadas")
with kpi4:
    acc_val = metricas_ref.get("acuracia", 0.860)
    st.metric("Acurácia em Teste", f"{acc_val * 100:.2f}%", f"{metricas_ref.get('amostras', 450)} amostras")

# ==============================================================================
# NAVEGAÇÃO EM ABAS DO DASHBOARD
# ==============================================================================
aba_simulador, aba_governanca, aba_analytics = st.tabs(
    [
        "🎯 1. Simulador Preditivo de Crédito (POST /predict)",
        "🛡️ 2. Laboratório de Validação & Erros HTTP (200 / 400 / 422)",
        "📊 3. Métricas do Pipeline & Exploração de Dados (Fase 1)",
    ]
)

# ------------------------------------------------------------------------------
# ABA 1: SIMULADOR PREDITIVO INTERATIVO
# ------------------------------------------------------------------------------
with aba_simulador:
    col_form, col_result = st.columns([1.05, 1.15], gap="large")

    with col_form:
        st.subheader("📋 Dados Cadastrais e Financeiros do Proponente")

        # Presets rápidos para agilizar demonstração em vídeo
        preset = st.radio(
            "Carregar Perfil de Demonstração Rápida:",
            options=[
                "Personalizado / Manual",
                "🟢 Perfil A: Baixo Risco (Alta Probabilidade de Aprovação)",
                "🟡 Perfil B: Risco Moderado (Análise Limítrofe)",
                "🔴 Perfil C: Alto Risco (Provável Reprovação)",
            ],
            horizontal=False,
        )

        if "🟢" in preset:
            defaults = dict(
                nome="Ana Beatriz Souza",
                idade=38,
                score=785,
                renda=12500.0,
                valor=18000.0,
                emprego=9.0,
                inadimplencia=0,
                moradia="PROPRIA",
                finalidade="NEGOCIOS",
            )
        elif "🟡" in preset:
            defaults = dict(
                nome="Marcos Vinícius Lima",
                idade=29,
                score=615,
                renda=4200.0,
                valor=19000.0,
                emprego=2.5,
                inadimplencia=0,
                moradia="FINANCIADA",
                finalidade="PESSOAL",
            )
        elif "🔴" in preset:
            defaults = dict(
                nome="Roberto Almeida",
                idade=22,
                score=410,
                renda=1900.0,
                valor=38000.0,
                emprego=0.5,
                inadimplencia=1,
                moradia="ALUGADA",
                finalidade="QUITACAO_DIVIDAS",
            )
        else:
            defaults = dict(
                nome="Carlos Eduardo Santos",
                idade=35,
                score=720,
                renda=7500.0,
                valor=18000.0,
                emprego=5.0,
                inadimplencia=0,
                moradia="PROPRIA",
                finalidade="NEGOCIOS",
            )

        with st.form("form_predicao_credito"):
            nome_cliente = st.text_input("Nome Completo do Solicitante", value=defaults["nome"])

            c1, c2 = st.columns(2)
            with c1:
                idade = st.slider(
                    "Idade (Anos)",
                    min_value=18,
                    max_value=100,
                    value=defaults["idade"],
                    help="Restrição Pydantic: ge=18, le=100",
                )
                renda_mensal = st.number_input(
                    "Renda Mensal Bruta (R$)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=defaults["renda"],
                    step=250.0,
                )
                tempo_emprego_anos = st.number_input(
                    "Tempo de Emprego Atual (Anos)",
                    min_value=0.0,
                    max_value=60.0,
                    value=defaults["emprego"],
                    step=0.5,
                )
                tipo_moradia = st.selectbox(
                    "Situação de Moradia",
                    options=["PROPRIA", "ALUGADA", "FINANCIADA", "OUTROS"],
                    index=["PROPRIA", "ALUGADA", "FINANCIADA", "OUTROS"].index(defaults["moradia"]),
                )

            with c2:
                pontuacao_credito = st.slider(
                    "Score de Crédito (300 a 850)",
                    min_value=300,
                    max_value=850,
                    value=defaults["score"],
                    help="Restrição Pydantic: ge=300, le=850",
                )
                valor_emprestimo = st.number_input(
                    "Valor Solicitado do Empréstimo (R$)",
                    min_value=500.0,
                    max_value=500000.0,
                    value=defaults["valor"],
                    step=500.0,
                )
                inadimplencia_previa = st.selectbox(
                    "Inadimplência nos Últimos 24 Meses?",
                    options=[0, 1],
                    format_func=lambda x: "0 — Não possui apontamentos" if x == 0 else "1 — Possui inadimplência prévia",
                    index=defaults["inadimplencia"],
                )
                finalidade_emprestimo = st.selectbox(
                    "Finalidade do Empréstimo",
                    options=["EDUCACAO", "SAUDE", "NEGOCIOS", "PESSOAL", "QUITACAO_DIVIDAS", "IMOVEL"],
                    index=["EDUCACAO", "SAUDE", "NEGOCIOS", "PESSOAL", "QUITACAO_DIVIDAS", "IMOVEL"].index(
                        defaults["finalidade"]
                    ),
                )

            submetido = st.form_submit_button(
                f"🚀 Executar Predição via FastAPI ({rota_escolhida})",
                use_container_width=True,
                type="primary",
            )

    with col_result:
        st.subheader("📈 Resultado da Inferência em Memória RAM")

        payload_envio = {
            "nome_cliente": nome_cliente,
            "idade": int(idade),
            "pontuacao_credito": int(pontuacao_credito),
            "renda_mensal": float(renda_mensal),
            "valor_emprestimo": float(valor_emprestimo),
            "tempo_emprego_anos": float(tempo_emprego_anos),
            "inadimplencia_previa": int(inadimplencia_previa),
            "tipo_moradia": tipo_moradia,
            "finalidade_emprestimo": finalidade_emprestimo,
        }

        if submetido or "ultima_resposta" not in st.session_state:
            status_code, resposta_json, latencia_pred = chamar_predict_api(
                api_base_url, rota_escolhida, payload_envio
            )
            st.session_state["ultima_resposta"] = (
                status_code,
                resposta_json,
                latencia_pred,
                payload_envio,
            )
            if submetido:
                st.session_state["historico_requisicoes"].insert(
                    0,
                    {
                        "Horário": time.strftime("%H:%M:%S"),
                        "Endpoint": rota_escolhida,
                        "Cliente": nome_cliente,
                        "Idade": idade,
                        "Score": pontuacao_credito,
                        "HTTP Status": status_code,
                        "Decisão": resposta_json.get("decisao", "Erro/Bloqueio"),
                        "Prob. Aprovação": (
                            f"{resposta_json.get('probabilidade_aprovacao', 0) * 100:.1f}%"
                            if status_code == 200
                            else "-"
                        ),
                        "Latência (ms)": f"{latencia_pred:.1f}",
                    },
                )

        status_code, resposta_json, latencia_pred, payload_usado = st.session_state["ultima_resposta"]

        if status_code == 200:
            decisao = resposta_json["decisao"]
            prob_aprov = resposta_json["probabilidade_aprovacao"]
            prob_risco = resposta_json["probabilidade_risco_inadimplencia"]
            faixa_risco = resposta_json["faixa_risco"]
            comprometimento = resposta_json["indice_comprometimento_anual"]

            if decisao == "Aprovado":
                st.markdown(
                    f"""
                    <div class="decision-card-approved">
                        <h3 style="margin:0;">✅ CRÉDITO APROVADO — {faixa_risco.upper()}</h3>
                        <p style="margin:0.3rem 0 0 0;">
                            Cliente: <b>{resposta_json.get('nome_cliente')}</b> •
                            Código Decisão: <code>{resposta_json['codigo_decisao']}</code> •
                            Status HTTP: <b>200 OK</b> ({latencia_pred:.1f} ms)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="decision-card-rejected">
                        <h3 style="margin:0;">❌ CRÉDITO REPROVADO — {faixa_risco.upper()}</h3>
                        <p style="margin:0.3rem 0 0 0;">
                            Cliente: <b>{resposta_json.get('nome_cliente')}</b> •
                            Código Decisão: <code>{resposta_json['codigo_decisao']}</code> •
                            Status HTTP: <b>200 OK</b> ({latencia_pred:.1f} ms)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.plotly_chart(
                criar_grafico_velocimetro(prob_aprov, decisao),
                use_container_width=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Prob. de Aprovação", f"{prob_aprov * 100:.2f}%")
            m2.metric("Risco de Inadimplência", f"{prob_risco * 100:.2f}%")
            m3.metric("Comprometimento Renda Anual", f"{comprometimento * 100:.1f}%")

        elif status_code == 400:
            st.error(f"🚫 **HTTP 400 Bad Request (Violação de Regra de Negócio):** {resposta_json.get('detail')}")
        elif status_code == 422:
            st.warning("⚠️ **HTTP 422 Unprocessable Entity (Validação Estrita Pydantic):**")
            st.json(resposta_json)
        else:
            st.error(f"Erro na comunicação HTTP ({status_code}): {resposta_json.get('detail')}")

        with st.expander("🔍 Inspecionar Contrato JSON (Request Enviado vs. Response Recebido)", expanded=False):
            j1, j2 = st.columns(2)
            with j1:
                st.markdown("**Payload JSON Enviado (`DadosClienteInput`):**")
                st.code(json.dumps(payload_usado, indent=2, ensure_ascii=False), language="json")
            with j2:
                st.markdown(f"**Resposta da API (`HTTP {status_code}`):**")
                st.code(json.dumps(resposta_json, indent=2, ensure_ascii=False), language="json")

    if st.session_state["historico_requisicoes"]:
        st.markdown("---")
        st.subheader("🕒 Histórico de Requisições Realizadas nesta Sessão")
        st.dataframe(pd.DataFrame(st.session_state["historico_requisicoes"]), use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 2: LABORATÓRIO DE VALIDAÇÃO & GOVERNANÇA HTTP (FASES 4 E 5)
# ------------------------------------------------------------------------------
with aba_governanca:
    st.subheader("🛡️ Demonstração Ao Vivo das Fases 4 e 5 (Pydantic & Semântica HTTP)")
    st.markdown(
        "Use os cenários abaixo durante a **gravação do seu vídeo avaliativo** para comprovar ao avaliador "
        "que a API FastAPI intercepta dados inválidos (`422`), bloqueia inconsistências de negócio (`400`) "
        "e mantém retrocompatibilidade com os arquivos base (`200 OK`)."
    )

    c_cen1, c_cen2, c_cen3 = st.columns(3, gap="medium")

    with c_cen1:
        st.markdown("#### 1️⃣ Contrato Mínimo Retrocompatível (`200 OK`)")
        st.caption(
            "Envia apenas `idade` e `pontuacao_credito` (como no arquivo `cliente.py` base). "
            "O schema `DadosClienteInput` preenche os demais atributos padrão e executa a inferência."
        )
        payload_c1 = {"idade": 40, "pontuacao_credito": 760}
        st.code(json.dumps(payload_c1, indent=2), language="json")
        if st.button("▶️ Testar Contrato Mínimo (200 OK)", use_container_width=True):
            sc, resp, lat = chamar_predict_api(api_base_url, rota_escolhida, payload_c1)
            st.success(f"Status HTTP: {sc} OK ({lat:.1f} ms)")
            st.json(resp)

    with c_cen2:
        st.markdown("#### 2️⃣ Violação de Limites Pydantic (`422 Unprocessable Entity`)")
        st.caption(
            "Envia `idade: 15` (menor que `ge=18`) e `pontuacao_credito: 990` (maior que `le=850`). "
            "O Pydantic bloqueia a entrada automaticamente antes de acionar o modelo."
        )
        payload_c2 = {"idade": 15, "pontuacao_credito": 990}
        st.code(json.dumps(payload_c2, indent=2), language="json")
        if st.button("▶️ Testar Bloqueio Pydantic (422)", use_container_width=True):
            sc, resp, lat = chamar_predict_api(api_base_url, rota_escolhida, payload_c2)
            st.warning(f"Status HTTP Retornado: {sc} Unprocessable Entity ({lat:.1f} ms)")
            st.json(resp)

    with c_cen3:
        st.markdown("#### 3️⃣ Inconsistência de Regra de Negócio (`400 Bad Request`)")
        st.caption(
            "Envia tipos válidos para o Pydantic (`idade: 20`), mas com `tempo_emprego_anos: 14.0` "
            "(impossível ter 14 anos de emprego formal aos 20 anos de idade)."
        )
        payload_c3 = {
            "idade": 20,
            "pontuacao_credito": 710,
            "renda_mensal": 4500.0,
            "valor_emprestimo": 10000.0,
            "tempo_emprego_anos": 14.0,
        }
        st.code(json.dumps(payload_c3, indent=2), language="json")
        if st.button("▶️ Testar Regra Financeira (400)", use_container_width=True):
            sc, resp, lat = chamar_predict_api(api_base_url, rota_escolhida, payload_c3)
            st.error(f"Status HTTP Retornado: {sc} Bad Request ({lat:.1f} ms)")
            st.json(resp)

# ------------------------------------------------------------------------------
# ABA 3: ANALYTICS & PERFORMANCE DO PIPELINE DE ML (FASE 1)
# ------------------------------------------------------------------------------
with aba_analytics:
    st.subheader("📊 Relatório Oficial de Métricas do Pipeline (`models/relatorio_metricas.json`)")

    if METRICS_PATH.exists():
        relatorio = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        metricas_particoes = relatorio.get("metricas", {})

        df_metricas = pd.DataFrame(
            [
                {
                    "Partição": m["particao"].capitalize(),
                    "Acurácia": m["acuracia"],
                    "Precisão": m["precisao"],
                    "Recall": m["recall"],
                    "F1-Score": m["f1_score"],
                    "ROC-AUC": m["roc_auc"],
                }
                for m in metricas_particoes.values()
            ]
        )

        g1, g2 = st.columns([1.2, 1.0], gap="large")
        with g1:
            df_long = df_metricas.melt(
                id_vars="Partição",
                value_vars=["Acurácia", "Precisão", "Recall", "F1-Score", "ROC-AUC"],
                var_name="Métrica",
                value_name="Valor",
            )
            fig_bar = px.bar(
                df_long,
                x="Métrica",
                y="Valor",
                color="Partição",
                barmode="group",
                text_auto=".3f",
                title="Comparativo de Desempenho: Treino (70%) vs. Validação (15%) vs. Teste (15%)",
                color_discrete_sequence=["#1e3a8a", "#0284c7", "#10b981"],
            )
            fig_bar.update_yaxes(range=[0.0, 1.08])
            st.plotly_chart(fig_bar, use_container_width=True)

        with g2:
            mc = metricas_particoes.get("teste", {}).get("matriz_confusao", [[0, 0], [0, 0]])
            fig_cm = px.imshow(
                mc,
                text_auto=True,
                labels=dict(x="Classe Predita pelo Modelo", y="Classe Real Observada", color="Amostras"),
                x=["Reprovado (0)", "Aprovado (1)"],
                y=["Reprovado (0)", "Aprovado (1)"],
                color_continuous_scale="Blues",
                title="Matriz de Confusão — Conjunto de Teste (15%)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Execute `python train_pipeline.py` para gerar o arquivo `models/relatorio_metricas.json`.")

    st.markdown("---")
    st.subheader("🔎 Exploração da Base de Dados Local (`data/credito_dataset.csv`)")
    if DATASET_PATH.exists():
        df_dados = pd.read_csv(DATASET_PATH)
        df_dados["Decisão"] = df_dados["aprovado"].map({1: "Aprovado (1)", 0: "Reprovado (0)"})

        d1, d2 = st.columns(2, gap="large")
        with d1:
            fig_hist = px.histogram(
                df_dados,
                x="pontuacao_credito",
                color="Decisão",
                nbins=35,
                barmode="overlay",
                opacity=0.75,
                title="Distribuição do Score de Crédito por Decisão",
                color_discrete_map={"Aprovado (1)": "#10b981", "Reprovado (0)": "#ef4444"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with d2:
            fig_box = px.box(
                df_dados,
                x="finalidade_emprestimo",
                y="valor_emprestimo",
                color="Decisão",
                title="Valor Solicitado por Finalidade e Status de Aprovação",
                color_discrete_map={"Aprovado (1)": "#10b981", "Reprovado (0)": "#ef4444"},
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with st.expander("📄 Visualizar Amostra da Base de Dados (`data/credito_dataset.csv`)"):
            st.dataframe(df_dados.head(50), use_container_width=True)
    else:
        st.info("Base `data/credito_dataset.csv` será gerada automaticamente ao rodar `python train_pipeline.py` ou iniciar a API.")
