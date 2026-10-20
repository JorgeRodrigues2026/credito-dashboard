# CreditAI MLOps — dashboard de apresentação

## Arquivos
```
index.html             página (gerada a partir de src/body.html)
styles.css             visual, tema claro/escuro, modo apresentação
script.js              Random Forest em JS, réplica da FastAPI, gráficos, interações
assets/model.js        as 200 árvores do modelo_emprestimo.joblib (scikit-learn 1.5.2)
assets/data.js         métricas, ROC, importâncias, taxas, latência e respostas reais da API
assets/st_*.webp       capturas do dashboard.py (Streamlit) rodando contra a API
src/body.html          conteúdo editável da página
build.py               gera index.html a partir de src/body.html
.nojekyll
```

## Abrir no computador
Dê duplo clique no `index.html`, ou rode `python -m http.server 5500` nesta pasta e abra http://localhost:5500.

## Apresentar
Clique em **Apresentar** (ou tecle **P**). Use as setas ou a barra de espaço para avançar, **Esc** para sair.

## Publicar no GitHub Pages
1. Envie todo o conteúdo desta pasta para um repositório (ex.: `credito-dashboard`).
2. GitHub → Settings → Pages → Deploy from a branch → `main` / root → Save.
3. Endereço: https://jorgerodrigues2026.github.io/credito-dashboard

## Ligar o simulador na API real
1. No `main.py`, depois de `app = FastAPI(...)`, adicione:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
```
2. `uvicorn main:app --reload`
3. No simulador, escolha "API FastAPI real" → `http://127.0.0.1:8000` → Conectar.

## Se treinar o modelo de novo
`assets/model.js` e `assets/data.js` foram gerados a partir do modelo atual. Um modelo re-treinado
precisa ser exportado de novo para o simulador continuar idêntico à API.
