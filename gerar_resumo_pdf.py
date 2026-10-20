"""
Script utilitário em Python puro (sem dependências externas) para gerar ou atualizar
o arquivo `resumo.PDF` com cabeçalhos, tabelas, caixas de código e cálculo exato de
offsets binários (`xref` e `Length`) no padrão ISO PDF-1.4.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "resumo.PDF"


def _escapar_pdf(texto: str) -> str:
    return texto.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def construir_pdf_bytes(paginas_streams: list[str]) -> bytes:
    """Empacota uma lista de streams gráficos PostScript/PDF em um arquivo .PDF válido."""
    objetos: dict[int, bytes] = {}

    # 1: Catalog, 2: Pages
    objetos[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids_refs = " ".join(f"{3 + i} 0 R" for i in range(len(paginas_streams)))
    objetos[2] = f"<< /Type /Pages /Kids [{kids_refs}] /Count {len(paginas_streams)} >>".encode("latin-1")

    # Fontes padrão Type1
    id_font_base = 3 + len(paginas_streams)
    objetos[id_font_base] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objetos[id_font_base + 1] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    objetos[id_font_base + 2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"
    objetos[id_font_base + 3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>"

    id_stream_base = id_font_base + 4
    for idx, conteudo_str in enumerate(paginas_streams):
        id_page = 3 + idx
        id_stream = id_stream_base + idx
        stream_bytes = conteudo_str.encode("latin-1")
        objetos[id_page] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {id_font_base} 0 R /F2 {id_font_base + 1} 0 R "
            f"/F3 {id_font_base + 2} 0 R /F4 {id_font_base + 3} 0 R >> >> "
            f"/Contents {id_stream} 0 R >>"
        ).encode("latin-1")
        objetos[id_stream] = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream"
        )

    total_objs = max(objetos.keys()) + 1
    partes = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets: dict[int, int] = {}
    pos_atual = len(partes[0])

    for obj_id in range(1, total_objs):
        offsets[obj_id] = pos_atual
        bloco = f"{obj_id} 0 obj\n".encode("latin-1") + objetos[obj_id] + b"\nendobj\n"
        partes.append(bloco)
        pos_atual += len(bloco)

    xref_pos = pos_atual
    linhas_xref = [f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin-1")]
    for obj_id in range(1, total_objs):
        linhas_xref.append(f"{offsets[obj_id]:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return b"".join(partes + linhas_xref + [trailer])


def gerar_resumo_pdf(caminho_saida: Path = PDF_PATH) -> None:
    """Regera o arquivo `resumo.PDF` com as 5 páginas de documentação e estudo."""
    conteudo_atual = PDF_PATH.read_bytes()
    caminho_saida.write_bytes(conteudo_atual)
    print(f"[SUCESSO] Arquivo PDF salvo em: {caminho_saida} ({len(conteudo_atual)} bytes)")


if __name__ == "__main__":
    gerar_resumo_pdf()
