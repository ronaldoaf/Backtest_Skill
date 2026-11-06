"""Valida a skill e, com --zip, gera o pacote para enviar ao Claude.ai.

    python ferramentas/validar_skill.py          # só valida
    python ferramentas/validar_skill.py --zip    # valida e gera dist/backtest.zip

Confere:
- o cabeçalho do SKILL.md (name em minúsculas com hífens, até 64 caracteres; description até
  1024 caracteres, sem < ou >);
- que todo arquivo citado (references/*.md, scripts/*.py) existe e que todo arquivo de
  references/ e scripts/ aparece na tabela de arquivos do SKILL.md;
- que todo nome em `snake_case` citado entre crases nos .md existe no código dos scripts
  (função, parâmetro ou chave de resultado), para a documentação não citar nada que não exista.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SKILL = RAIZ / "skills" / "backtest"

# Nomes entre crases que são código de exemplo do usuário, não da skill.
FORA_DO_CODIGO = {"p_nova", "p_antiga", "retorno_estrategia", "retorno_liquido", "custo_por_lado",
                  "capital_final", "capital_inicial", "periodos_por_ano", "cdi_anual",
                  "ganho_médio", "perda_média"}


def cabecalho(texto: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", texto, re.S)
    if not m:
        raise SystemExit("SKILL.md sem cabeçalho YAML (--- ... ---).")
    campos = {}
    for linha in m.group(1).splitlines():
        chave, _, valor = linha.partition(":")
        campos[chave.strip()] = valor.strip()
    return campos


def validar() -> list[str]:
    erros = []
    skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    cab = cabecalho(skill_md)
    nome, desc = cab.get("name", ""), cab.get("description", "")
    if not re.fullmatch(r"[a-z0-9-]{1,64}", nome):
        erros.append(f"name inválido: {nome!r}")
    if not desc or len(desc) > 1024 or "<" in desc or ">" in desc:
        erros.append(f"description vazia, com mais de 1024 caracteres ({len(desc)}) ou com < >")

    docs = {p: p.read_text(encoding="utf-8") for p in [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]}
    for p, txt in docs.items():
        for ref in sorted(set(re.findall(r"(?:references|scripts)/[\w.-]+\.(?:md|py)", txt))):
            if not (SKILL / ref).exists():
                erros.append(f"{p.name} cita {ref}, que não existe")
    for arq in [*sorted((SKILL / "references").glob("*.md")), *sorted((SKILL / "scripts").glob("*.py"))]:
        rel = f"{arq.parent.name}/{arq.name}"
        if f"`{rel}`" not in skill_md:
            erros.append(f"{rel} não aparece na tabela de arquivos do SKILL.md")

    codigo = "\n".join(p.read_text(encoding="utf-8") for p in (SKILL / "scripts").glob("*.py"))
    palavras = set(re.findall(r"\w+", codigo))
    for p, txt in docs.items():
        for trecho in re.findall(r"`([^`\n]+)`", txt):
            nome_cod = re.match(r"([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?:\(|=|$|\[)", trecho)
            if nome_cod and nome_cod.group(1) not in palavras and nome_cod.group(1) not in FORA_DO_CODIGO:
                erros.append(f"{p.name} cita `{nome_cod.group(1)}`, que não existe nos scripts")
    return sorted(set(erros))


def empacotar() -> Path:
    destino = RAIZ / "dist" / "backtest.zip"
    destino.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for arq in sorted(SKILL.rglob("*")):
            if arq.is_file() and "__pycache__" not in arq.parts and arq.suffix != ".pyc":
                z.write(arq, Path("backtest") / arq.relative_to(SKILL))
    return destino


if __name__ == "__main__":
    erros = validar()
    for e in erros:
        print("ERRO:", e)
    if erros:
        sys.exit(1)
    print("Skill válida.")
    if "--zip" in sys.argv:
        print("Pacote:", empacotar().relative_to(RAIZ))
