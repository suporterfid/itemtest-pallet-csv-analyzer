# ItemTestAnalyzer

Automação de análises de CSVs do **Impinj ItemTest** com suporte a **layout opcional do pallet** (CSV/XLSX/Markdown).

## Requisitos
- Python 3.11+
- `pip install -r requirements.txt`

## Uso (CLI)
```
python src/analisar_itemtest.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Resultados" --layout "C:\RFID\Pallets\Layout.xlsx"
```
Sem layout:
```
python src/analisar_itemtest.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Resultados"
```

## Saídas
- Um `.xlsx` por CSV processado
- Gráficos em `output/graficos/<nome-arquivo>/`
- Logs simples em `output/logs/` (reservado para futura expansão)

## Notas
- Linhas em que `EPC` é IP (ex.: `192.168.68.100`) são ignoradas.
- EPCs inválidos (não hexadecimais/curtos) são removidos.
- Com **layout** fornecido, é gerada uma aba `Posicoes_Pallet` e `expected_suffix`.
