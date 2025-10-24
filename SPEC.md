# 📘 SPEC.md — Impinj ItemTest RFID Analyzer

*(suporte a testes estruturados e contínuos sem identificação prévia de tags)*

---

## 🎯 Objetivo

Desenvolver uma **aplicação Python completa** para análise automatizada de arquivos CSV gerados pelo **Impinj ItemTest**, usada em dois cenários de leitura RFID UHF:

1. **Cenário Estruturado (com layout de referência)** — testes controlados de leitura de totes/pallets, em que as posições dos tags são previamente conhecidas.
2. **Cenário de Campo (sem identificação prévia)** — leitura contínua de tags durante o **descarregamento de múltiplos pallets** a partir de uma **carreta na doca de um Centro de Distribuição (CD)**, sem arquivo de referência prévio e com alto volume de EPCs novos a cada sessão.

A aplicação deve detectar o tipo de cenário automaticamente ou aceitar parâmetros explícitos (`--mode structured` / `--mode continuous`).

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

  * EPCs inválidos (não-hexadecimais ou curtos);
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

* Correlacionar EPCs ou sufixos com posições físicas do pallet.
* Calcular cobertura e falhas de leitura por posição.
* Gerar indicadores executivos e técnicos:

  * **CoverageRate** = % de EPCs esperados lidos.
  * **AntennaBalance** = variação percentual entre antenas.
  * **TopPerformerAntenna** = antena com maior número de leituras.
  * **RSSI Stability Index** = desvio padrão do RSSI médio entre antenas.
* Associar colunas `Posição_Pallet`, `Linha`, `Face`, etc.

---

### 4. Modo Contínuo (sem identificação prévia)

Quando **nenhum arquivo de layout** for fornecido ou `--mode continuous` for informado:

1. **Análise de fluxo contínuo:**

   * Agrupar leituras por EPC e janelas de tempo (`--window`, padrão 2 segundos).
   * Detectar **entrada** e **saída** de EPCs (EPC ausente > janela → saiu).
   * Calcular:

     * `duration_present` = tempo total de permanência.
     * `read_events` = número de ciclos de entrada/saída.
     * `antenna_distribution` = % de leituras por antena.
     * `direction_estimate` = estimativa de sentido de movimento (ex.: 1→3 = entrada).
     * `RSSI_variability` = desvio padrão de RSSI do EPC.
     * `active_concurrency` = nº médio de EPCs simultâneos por segundo.
   * Detectar **leituras anômalas**:

     * EPCs com duração longa demais → possível obstrução.
     * EPCs detectados em antenas incompatíveis (superiores apenas, etc.).

2. **Métricas executivas agregadas:**

   * EPCs distintos por minuto.
   * Duração média no campo.
   * Taxa de leitura contínua (% do tempo com EPCs ativos).
   * Antena dominante (maior volume de leituras).
   * RSSI médio global e variação.

3. **Agrupamento por sessão:**

   * Separar períodos ativos por tempo sem leitura.
   * Calcular KPIs por sessão:

     * Duração da sessão.
     * EPCs únicos detectados.
     * Throughput médio (EPCs/minuto).
     * Antena dominante.
     * RSSI médio por sessão.

---

## 📊 Métricas Complementares (Executivas e Técnicas)

| Métrica                  | Descrição                                    | Aplicável a | Tipo      |
| ------------------------ | -------------------------------------------- | ----------- | --------- |
| `CoverageRate`           | % de EPCs esperados lidos                    | Structured  | Executivo |
| `TotalDistinctEPCs`      | Número de EPCs únicos lidos                  | Ambos       | Executivo |
| `AverageRSSI`            | Média geral de RSSI                          | Ambos       | Técnico   |
| `RSSI_StdDev`            | Desvio padrão de RSSI (estabilidade)         | Ambos       | Técnico   |
| `BestRSSI` / `WorstRSSI` | Melhores/piores leituras por EPC             | Ambos       | Técnico   |
| `AntennaParticipation`   | % de leituras por antena                     | Ambos       | Técnico   |
| `AntennaBalance`         | Desbalanceamento entre antenas               | Ambos       | Técnico   |
| `TagReadRedundancy`      | Leituras repetidas do mesmo EPC              | Ambos       | Técnico   |
| `TagDwellTimeAvg`        | Tempo médio de permanência no campo          | Continuous  | Executivo |
| `TagDwellTimeMax`        | Maior tempo de permanência detectado         | Continuous  | Técnico   |
| `ConcurrentTagsPeak`     | Máx. de EPCs simultâneos ativos              | Continuous  | Técnico   |
| `ReadContinuityRate`     | % de tempo com EPCs sendo detectados         | Continuous  | Executivo |
| `ThroughputPerMinute`    | EPCs distintos/minuto                        | Continuous  | Executivo |
| `SessionDuration`        | Duração total da sessão de leitura           | Continuous  | Executivo |
| `DirectionEstimate`      | Sentido de passagem do EPC                   | Continuous  | Técnico   |
| `FrequencyUsage`         | Faixas de frequência mais utilizadas         | Ambos       | Técnico   |
| `ModePerformance`        | Comparativo de ModeIndex vs. taxa de leitura | Ambos       | Técnico   |
| `NoiseIndicator`         | RSSI alto sem EPC → possível interferência   | Ambos       | Técnico   |
| `Total de Cajas Leydo`                 | Quantidade de totes/caixas com leitura válida         | Structured  | Executivo |
| `Tasa promedio de lectura por intento` | Conversão média de tentativas de leitura em sucesso   | Ambos       | Executivo |
| `Tiempo promedio de lectura por tote`  | Tempo médio necessário para completar a leitura de cada tote | Structured  | Executivo |
| `Tasa de fallas de leitura`            | % de tentativas de leitura que não retornam EPC válido | Ambos       | Técnico   |
| `Tasa de lecturas duplicadas`          | % de eventos redundantes sobre o total de leituras    | Ambos       | Técnico   |
| `Cobertura del área de leitura`        | % da área/posições do layout efetivamente coberta     | Structured  | Executivo |
| `Capacidad de lectura simultánea`      | Nº médio de totes/pallets lidos ao mesmo tempo        | Continuous  | Executivo |
| `Disponibilidad del sistema`           | % de tempo com leitor operacional e registrando dados | Ambos       | Executivo |

Cada métrica deve ser apresentada nos relatórios executivos com a estrutura tabular:

* **Indicador** — nome do KPI conforme listado acima (ex.: “Cobertura del área de leitura”).
* **Resultado** — valor calculado no período/sessão (percentual, contagem ou tempo médio) com unidades explícitas.
* **Interpretação executiva** — breve leitura gerencial do resultado, incluindo alertas ou metas de referência.

Esse formato se aplica aos dashboards textuais, à aba `Indicadores_Executivos` do Excel e a quaisquer exportações resumidas.

---

## 📈 Saídas Esperadas

### 1. **Planilha Excel (por teste/sessão)**

Abas:

* `Resumo_por_EPC` — estatísticas individuais.
* `Leituras_por_Antena` — desempenho detalhado por antena.
* `Fluxo_Contínuo` — entradas/saídas, duração e eventos (modo contínuo).
* `Indicadores_Executivos` — KPIs de performance geral.
  * Deve listar explicitamente: `CoverageRate`, `TotalDistinctEPCs`, `Total de Cajas Leydo`, `Tasa promedio de lectura por intento`, `Tiempo promedio de lectura por tote`, `Tasa de fallas de leitura`, `Tasa de lecturas duplicadas`, `Cobertura del área de leitura`, `Capacidad de lectura simultánea`, `Disponibilidad del sistema`, além de `AverageRSSI`, `RSSI_StdDev`, `AntennaBalance`, `TagReadRedundancy`, `TagDwellTimeAvg`, `ConcurrentTagsPeak`, `ReadContinuityRate`, `ThroughputPerMinute`, `SessionDuration`, `ModePerformance` e `NoiseIndicator`.
* `Metadata` — parâmetros de teste (RF mode, potência, etc.).
* `Posicoes_Pallet` — cobertura física (modo estruturado).

### 2. **Gráficos automáticos**

* Barras — Leituras por EPC.
* Barras — Leituras por Antena.
* Boxplot — RSSI por Antena.
* Linha — EPCs ativos ao longo do tempo (modo contínuo).
* Heatmap — cobertura ou intensidade (modo contínuo/estruturado).
* Linha — Throughput (EPCs/min).
* Dispersão — RSSI vs Frequência (detecção de ruído).

### 3. **Resumo Textual Automático**

> “Durante o descarregamento monitorado entre 14:00 e 14:15,
> o leitor `192.168.68.100` detectou **1.280 EPCs distintos**,
> tempo médio de permanência **3,4s**, RSSI médio **–53,2 dBm**,
> `Capacidad de lectura simultánea` **2,4 totes**, `Tasa de fallas de leitura` **1,2%**,
> e **Antena 3** responsável por **47% das leituras**.”

Os resumos textuais e dashboards devem aproveitar a mesma lista de indicadores executivos, garantindo que `Total de Cajas Leydo`, `Tasa promedio de lectura por intento`, `Tiempo promedio de lectura por tote`, `Tasa de fallas de leitura`, `Tasa de lecturas duplicadas`, `Cobertura del área de leitura`, `Capacidad de lectura simultánea` e `Disponibilidad del sistema` sejam destacados sempre que houver dados disponíveis.

---

## ⚙️ Requisitos Técnicos

* **Linguagem:** Python 3.11+
* **Bibliotecas:** `pandas`, `numpy`, `matplotlib`, `xlsxwriter`, `argparse`, `pathlib`
* **Execução CLI:**

  ```bash
  python itemtest_analyzer.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Resultados" --mode continuous --window 2
  ```

  ou

  ```bash
  python itemtest_analyzer.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Resultados" --layout "C:\RFID\Pallets\Layout.xlsx" --mode structured
  ```
* **Extras:**

  * `requirements.txt`
  * `run.bat` (execução simplificada)
  * Logs detalhados (`/output/logs/`)
  * Suporte a parâmetros adicionais:

    * `--window <segundos>` → define janela temporal para detecção de saída.
    * `--summary` → gera relatório executivo consolidado.

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
│   ├── continuous_mode.py
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

## 💡 Visualização (Streamlit ou Power BI)

Se habilitado:

* Upload de múltiplos CSVs e layouts.
* Exibição em **cards**:

  * “EPCs distintos: 1.280”
  * “Tempo médio de leitura: 3,4s”
  * “RSSI médio: –53 dBm”
  * “Antena dominante: 3”
* Gráficos interativos (timeline, heatmap, antenas).
* Exportação em Excel e PDF consolidado.

---

## 🧩 Tarefas do Agente

1. Adicionar métricas complementares na camada `metrics.py`.
2. Implementar agrupamento temporal e janelas móveis no `continuous_mode.py`.
3. Atualizar relatórios e gráficos para suportar novos indicadores.
4. Garantir retrocompatibilidade com estrutura existente.
5. (Opcional) Adicionar painel interativo em Streamlit/Power BI.

---

## 📘 Exemplo de Saída Consolidada

| EPC           | Leituras | RSSI_médio | RSSI_StdDev | Duração (s) | Entradas | Saídas | Ant_Principal | Concurrency_Peak |
| ------------- | -------- | ---------- | ----------- | ----------- | -------- | ------ | ------------- | ---------------- |
| 3008_33...B6E | 124      | -54.2      | 4.5         | 3.2         | 1        | 1      | 3             | 87               |
| 3008_33...2FA | 118      | -59.7      | 3.8         | 5.8         | 1        | 1      | 4             | 65               |
| 3008_33...9F3 | 16       | -67.1      | 6.9         | 0.8         | 1        | 0      | 2             | 12               |

---

