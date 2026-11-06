"""A documentação só cita funções e arquivos que existem, e o pacote da skill é gerado."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ferramentas"))

import validar_skill  # noqa: E402


def test_skill_valida():
    assert validar_skill.validar() == []


def test_pacote_tem_skill_md_na_raiz_da_pasta():
    destino = validar_skill.empacotar()
    with zipfile.ZipFile(destino) as z:
        nomes = z.namelist()
    assert "backtest/SKILL.md" in nomes
    assert "backtest/scripts/backtest.py" in nomes
    assert not any("__pycache__" in n for n in nomes)


if __name__ == "__main__":
    test_skill_valida()
    test_pacote_tem_skill_md_na_raiz_da_pasta()
    print("2 testes passaram")
