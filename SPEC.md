# Specification — Impinj ItemTest RFID Analyzer

*(suporte a testes estruturados e contínuos sem identificação prévia de tags)*

## 🎯 Objetivo

Desenvolver uma **aplicação Python completa** para análise automatizada de arquivos CSV gerados pelo **Impinj ItemTest**, usada em dois cenários de leitura RFID UHF:

1. **Cenário Estruturado (com layout de referência)** — testes controlados de leitura de totes/pallets, em que as posições dos tags são previamente conhecidas.
2. **Cenário de Campo (sem identificação prévia)** — leitura contínua de tags durante o **descarregamento de múltiplos pallets** a partir de uma **carreta na doca de um Centro de Distribuição (CD)**, sem arquivo de referência prévio e com alto volume de EPCs novos a cada sessão.

A aplicação deve detectar o tipo de cenário automaticamente ou aceitar parâmetros explícitos de modo de operação (`--mode structured` / `--mode continuous`).

---

## 📂 Entradas

### 1. Arquivos CSV do Impinj ItemTest

* Localizados em uma pasta informada pelo usuário (ex.: `C:\RFID\Tests\ItemTest\CSV\`).
* Cada arquivo contém comentários iniciais (`//`) com metadados e cabeçalho real na 2ª ou 3ª linha:

  ```text
  // Antennas connected = 1,2,3,4, ModeIndex=1002, PowersInDbm=3=>17, 4=>15, Session=1, InventoryMode=DualTarget
  // Timestamp, EPC, TID, Antenna, RSSI, Frequency, Hostname, PhaseAngle, DopplerFrequency, CRHandle
  ```

### 2. Arquivo opcional de referência (modo estruturado)

* Contém os EPCs (completos ou parciais) e suas posições físicas no pallet.
* Formato aceito: **CSV**, **XLSX**, ou **Markdown (.md)**.
* Exemplo:

  ```text
  Linha,Traseira,Lateral_Esquerda,Lateral_Direita,Frente
  5,8FA,E77 / 8FA,,
  4,3D9 / B6E,E66 / 4DF / 3D9,B6E / 780,E66 / 780
  3,2FA / 827,A4B / 56B / 2EA,827 / 4F5,A4B / 4F5
  2,68C / 929,392 / F63 / 68C,929 / 052,392 / 052
  1,D6A / C38,D6A / B86 / 563,C38 / EE8,563 / EE8
  ```
* Se o arquivo não for fornecido, a aplicação funcionará em modo **sem referência** (campo livre).

---

## 🧠 Lógica de Processamento

### 1. Leitura e limpeza dos dados

* Ignorar linhas com:

  * EPCs inválidos (não hexadecimais ou curtos);
  * Endereços IP (`192.168.x.x`) na coluna `EPC`.
* Converter:

  * `RSSI`, `Antenna`, `Frequency`, `PhaseAngle`, `DopplerFrequency` → numéricas;
  * `Timestamp` → datetime.

---

### 2. Extração de metadados

Capturar do cabeçalho:

* `AntennaIDs`
* `ModeIndex`
* `PowersInDbm`
* `Session`
* `InventoryMode`
* `Hostname` (IP do leitor)

---

### 3. Modo Estruturado (com referência)

Aplicar a lógica existente:

* Correlacionar EPCs ou sufixos com posições físicas do pallet.
* Gerar análises de cobertura: quantos EPCs esperados foram lidos, quais faltaram.
* Associar colunas `Posição_Pallet`, `Linha`, `Face`, etc.

---

### 4. Modo Contínuo (sem identificação prévia)

Quando **nenhum arquivo de layout for fornecido** **ou** o parâmetro `--mode continuous` for informado:

1. **Analisar comportamento em fluxo contínuo:**

   * Agrupar leituras por EPC e intervalo temporal (ex.: janelas de 5 segundos).
   * Detectar “entradas” e “saídas” de EPCs do campo de leitura com base no tempo entre leituras.

     * Exemplo: EPC ausente por >2 s → considerado “saído”.
   * Calcular estatísticas agregadas:

     * `duration_present` = tempo total em que o EPC foi detectado.
     * `read_events` = número de eventos (entradas + saídas).
     * `antenna_distribution` = porcentagem de leituras por antena.

2. **Detectar padrões de movimento:**

   * Determinar direção do fluxo (ex.: Antena 1 → Antena 3 = entrada; Antena 4 → Antena 2 = saída).
   * Gerar alertas de inconsistência (ex.: EPC detectado apenas em antenas superiores = leitura parcial).

3. **Gerar relatórios resumidos:**

   * EPCs distintos detectados por minuto.
   * Tempo médio de permanência no campo de leitura.
   * EPCs que permaneceram por tempo anormalmente longo (potenciais bloqueios de leitura).

4. **Agrupar resultados por sessão (arquivo):**

   * Identificar automaticamente sessões de descarregamento contínuo.
   * Calcular desempenho médio das antenas por sessão.

---

## 📊 Saídas Esperadas

### 1. **Planilha Excel (por teste ou por sessão)**

Abas:

* `Resumo_por_EPC` — todas as leituras com estatísticas.
* `Leituras_por_Antena` — performance de antenas.
* `Fluxo_Contínuo` — (modo contínuo) entrada/saída e duração por EPC.
* `Metadata`
* `Posicoes_Pallet` — (somente se layout presente)

### 2. **Gráficos automáticos**

* Barras — Leituras por EPC
* Barras — Leituras por Antena
* Boxplot — RSSI por Antena
* Linha — EPCs ativos ao longo do tempo (modo contínuo)
* Heatmap — cobertura de antenas (modo contínuo)

### 3. **Sumário Textual Automático**

> “Durante o descarregamento monitorado entre 14:00 e 14:15,
> o leitor `192.168.68.100` detectou 1.280 EPCs distintos,
> com tempo médio de permanência de 3,4 segundos e RSSI médio de –53,2 dBm.
> A Antena 3 foi responsável por 47% das leituras totais.”

---

## ⚙️ Requisitos Técnicos

* **Linguagem:** Python 3.11+

* **Bibliotecas:** `pandas`, `numpy`, `matplotlib`, `xlsxwriter`, `argparse`, `pathlib`

* **Execução CLI:**

  ```bash
  python itemtest_analyzer.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Resultados" --mode continuous
  ```

  ou

  ```bash
  python itemtest_analyzer.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Resultados" --layout "C:\RFID\Pallets\Layout01.xlsx" --mode structured
  ```

* **Extras:**

  * `requirements.txt`
  * `run.bat` para execução no Windows
  * Logs em `output\logs\` com resumo de desempenho por antena e EPCs ativos por minuto.

---

## 🧱 Estrutura do Projeto

```
ItemTestAnalyzer/
├── src/
│   ├── itemtest_analyzer.py
│   ├── parser.py
│   ├── metrics.py
│   ├── plots.py
│   ├── pallet_layout.py
│   ├── continuous_mode.py   ← novo módulo para fluxo contínuo
│   └── report.py
├── output/
│   ├── figures/
│   └── logs/
├── samples/
│   ├── Sample_ItemTest.csv
│   └── Sample_Pallet_Layout.xlsx
├── requirements.txt
└── run.bat
```

---

## 💡 Visualização (opcional com Streamlit)

Se o agente suportar interface visual:

* Upload de múltiplos CSVs (testes ou sessões).
* Cards:

  * “EPCs distintos: 1.280”
  * “Média RSSI: –53 dBm”
  * “Duração média no campo: 3,4 s”
  * “Antena dominante: 3 (Base Esquerda)”
* Gráficos interativos com linha de tempo e clusters de EPCs por antena.
* Botão para baixar Excel consolidado.

---

## 🧩 Tarefas do Agente

1. Criar e atualizar scripts conforme os dois modos (`structured` e `continuous`).
2. Adicionar o novo módulo `continuous_mode.py`.
3. Criar agregações temporais (janelas de tempo configuráveis).
4. Gerar relatórios e gráficos automáticos.
5. Garantir compatibilidade total com Windows 11.
6. Adicionar suporte CLI para `--mode`.
7. (Opcional) Interface Streamlit para operação em tempo real.

---

## 📘 Exemplo de Saída Consolidada (modo contínuo)

| EPC           | Leituras | RSSI_médio | Duração (s) | Entradas | Saídas | Ant_Principal | Ant_1ª | Ant_Última |
| ------------- | -------- | ---------- | ----------- | -------- | ------ | ------------- | ------ | ---------- |
| 3008_33...B6E | 124      | -54.2      | 3.2         | 1        | 1      | 3             | 3      | 3          |
| 3008_33...2FA | 118      | -59.7      | 5.8         | 1        | 1      | 4             | 4      | 4          |
| 3008_33...9F3 | 16       | -67.1      | 0.8         | 1        | 0      | 2             | 2      | —          |


