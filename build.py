"""Gera index.html (GitHub Pages) e creditai.html (página para publicar) a partir de src/body.html."""
from pathlib import Path
body = Path("src/body.html").read_text(encoding="utf-8")
head_common = '''<title>CreditAI MLOps</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Apresentação interativa do microsserviço de concessão de crédito: modelo Random Forest, API FastAPI, validação Pydantic e painel Streamlit.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="styles.css">'''
scripts = '''
<script src="assets/model.js"></script>
<script src="assets/data.js"></script>
<script src="script.js"></script>'''
icon = '<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Crect width=%2732%27 height=%2732%27 rx=%278%27 fill=%27%230f6b4f%27/%3E%3C/svg%3E">'
Path("index.html").write_text(f'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head_common}
{icon}
</head>
<body>
{body}{scripts}
</body>
</html>
''', encoding="utf-8")
Path("creditai.html").write_text(f"{head_common}\n{body}{scripts}\n", encoding="utf-8")
print("index.html e creditai.html gerados")
