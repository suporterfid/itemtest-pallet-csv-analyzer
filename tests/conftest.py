# -*- coding: utf-8 -*-
"""Configura o ambiente de testes para permitir imports do pacote src."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
