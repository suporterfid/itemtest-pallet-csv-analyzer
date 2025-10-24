"""Utilities for assembling executive KPI tables and interpretations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import pandas as pd


CANONICAL_KPI_ORDER: tuple[str, ...] = (
    "CoverageRate",
    "TotalDistinctEPCs",
    "Total de Cajas Leydo",
    "Tasa promedio de lectura por intento",
    "Tiempo promedio de leitura por tote",
    "Tasa de fallas de leitura",
    "Tasa de lecturas duplicadas",
    "Cobertura del área de leitura",
    "Capacidad de lectura simultánea",
    "Disponibilidad del sistema",
    "AverageRSSI",
    "RSSI_StdDev",
    "AntennaBalance",
    "TagReadRedundancy",
    "TagDwellTimeAvg",
    "ConcurrentTagsPeak",
    "ReadContinuityRate",
    "ThroughputPerMinute",
    "SessionDuration",
    "ModePerformance",
    "NoiseIndicator",
)

CANONICAL_KPI_SET = set(CANONICAL_KPI_ORDER)


@dataclass
class ExecutiveEntry:
    """Simple container describing an executive KPI entry."""

    indicator: str
    result: str
    interpretation: str

    def as_row(self) -> dict[str, str]:
        """Return a dictionary representing the row to persist to Excel."""

        return {
            "Indicador": self.indicator,
            "Resultado": self.result,
            "Interpretação executiva": self.interpretation,
        }


def _is_missing(value: object) -> bool:
    """Return ``True`` when the supplied value should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, (pd.Series, pd.Index)):
        return value.empty
    return False


def _format_percentage(value: float, *, decimals: int = 2) -> str:
    """Format a numeric value as a percentage string."""

    return f"{float(value):.{decimals}f}%"


def _format_ratio(value: float, *, suffix: str = "×", decimals: int = 2) -> str:
    """Format a numeric ratio with the provided suffix."""

    return f"{float(value):.{decimals}f}{suffix}"


def _format_seconds(value: float) -> str:
    """Format seconds in a human-readable form keeping short activities granular."""

    seconds = float(value)
    if seconds >= 3600:
        hours = int(seconds // 3600)
        remainder = seconds % 3600
        minutes = int(remainder // 60)
        secs = int(round(remainder % 60))
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if seconds >= 300:
        minutes = int(seconds // 60)
        remainder = int(round(seconds % 60))
        return f"{minutes:02d}:{remainder:02d}"
    return f"{seconds:.2f} s"


def _format_timestamp(value: object) -> str:
    """Return a short timestamp label if ``value`` is datetime-like."""

    if value is None:
        return ""
    try:
        ts = pd.to_datetime(value)
    except Exception:  # pragma: no cover - defensive fallback
        return str(value)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M")


def _interpret_percentage(
    value: float | None,
    *,
    context: str,
    high_is_good: bool = True,
) -> str:
    """Return a qualitative interpretation for percentage-based metrics."""

    if value is None or _is_missing(value):
        return f"Sem dados suficientes para {context.lower()}."
    numeric = float(value)
    if high_is_good:
        if numeric >= 90:
            qualifier = "excelente"
        elif numeric >= 75:
            qualifier = "adequada"
        else:
            qualifier = "crítica"
    else:
        if numeric <= 5:
            qualifier = "dentro do esperado"
        elif numeric <= 15:
            qualifier = "em atenção"
        else:
            qualifier = "desbalanceada"
    return f"{context} {qualifier} ({numeric:.2f}%)."


def _interpret_variability(value: float | None, *, context: str) -> str:
    """Return qualitative assessment for variability metrics (lower is better)."""

    if value is None or _is_missing(value):
        return f"Sem dados suficientes sobre {context.lower()}."
    numeric = float(value)
    if numeric <= 1.5:
        qualifier = "estável"
    elif numeric <= 3.0:
        qualifier = "moderada"
    else:
        qualifier = "elevada"
    return f"{context} {qualifier} ({numeric:.2f})."


def _interpret_balance(value: float | None) -> str:
    """Return qualitative text for antenna balance percentage."""

    if value is None or _is_missing(value):
        return "Sem dados suficientes para avaliar o balanceamento de antenas."
    numeric = float(value)
    if numeric <= 5:
        qualifier = "balanceado"
    elif numeric <= 12:
        qualifier = "com leve assimetria"
    else:
        qualifier = "desbalanceado"
    return f"Balanceamento de antenas {qualifier} ({numeric:.2f}%)."


def _interpret_redundancy(value: float | None) -> str:
    """Return qualitative text for tag read redundancy."""

    if value is None or _is_missing(value):
        return "Sem dados suficientes para avaliar redundância de leitura."
    numeric = float(value)
    if numeric >= 2.0:
        qualifier = "saudável"
    elif numeric >= 1.2:
        qualifier = "limitada"
    else:
        qualifier = "insuficiente"
    return f"Redundância de leitura {qualifier} ({numeric:.2f}×)."


def _interpret_rssi(value: float | None) -> str:
    """Return qualitative text for global RSSI averages."""

    if value is None or _is_missing(value):
        return "Sem dados de RSSI global disponíveis."
    numeric = float(value)
    if numeric >= -55:
        qualifier = "forte"
    elif numeric >= -65:
        qualifier = "estável"
    else:
        qualifier = "fraco"
    return f"Sinal médio {qualifier} ({numeric:.2f} dBm)."


def _interpret_dwell(value: float | None) -> str:
    """Return qualitative text for average dwell time."""

    if value is None or _is_missing(value):
        return "Sem dados de permanência média."
    numeric = float(value)
    if numeric <= 2.5:
        qualifier = "rápida"
    elif numeric <= 5.0:
        qualifier = "moderada"
    else:
        qualifier = "prolongada"
    return f"Permanência média {qualifier} ({numeric:.2f} s)."


def _interpret_concurrency(value: float | None, average: float | None) -> str:
    """Return qualitative text for concurrency peaks."""

    if value is None or _is_missing(value):
        return "Sem dados de simultaneidade."
    peak = int(value)
    if average is None or _is_missing(average):
        return f"Pico de {peak} EPCs simultâneos."
    return f"Pico de {peak} EPCs simultâneos com média de {float(average):.2f}."


def _interpret_throughput(value: float | None) -> str:
    """Return qualitative text for throughput per minute."""

    if value is None or _is_missing(value):
        return "Sem dados de throughput por minuto."
    numeric = float(value)
    if numeric >= 40:
        qualifier = "alto"
    elif numeric >= 20:
        qualifier = "estável"
    else:
        qualifier = "reduzido"
    return f"Throughput {qualifier} ({numeric:.2f} EPCs/min)."


def _interpret_noise(
    indicator_text: str | None,
    *,
    noise_flag: bool | None,
    noise_reads: float | None,
) -> str:
    """Return a compact interpretation for the RSSI noise indicator."""

    if not indicator_text:
        return "Sem avaliação de ruído RSSI disponível."
    if noise_flag is None:
        status = "Avaliação qualitativa disponível."
    else:
        status = "Alerta de ruído detectado" if noise_flag else "RSSI dentro do esperado"
    detail = ""
    if noise_reads is not None and not _is_missing(noise_reads):
        detail = f" Leituras/EPC: {float(noise_reads):.2f}."
    return f"{status}.{detail}".strip()


def format_mode_indicator(source: dict | None) -> str | None:
    """Format the mode indicator text from structured or continuous metrics."""

    if not isinstance(source, dict):
        return None
    description = source.get("mode_performance_text")
    if description:
        return str(description)
    indicator = source.get("mode_performance")
    if isinstance(indicator, dict):
        description = indicator.get("description")
        if description:
            return str(description)
        label_parts: list[str] = []
        mode_value = indicator.get("mode_index")
        if mode_value is not None and not (
            isinstance(mode_value, float) and math.isnan(mode_value)
        ):
            label_parts.append(f"ModeIndex {mode_value}")
        rate_segments: list[str] = []
        reads_per_second = indicator.get("reads_per_second")
        if reads_per_second is not None and not pd.isna(reads_per_second):
            rate_segments.append(f"{float(reads_per_second):.2f} leituras/s")
        reads_per_minute = indicator.get("reads_per_minute")
        if reads_per_minute is not None and not pd.isna(reads_per_minute):
            rate_segments.append(f"{float(reads_per_minute):.2f} leituras/min")
        epcs_per_minute = indicator.get("epcs_per_minute")
        if epcs_per_minute is not None and not pd.isna(epcs_per_minute):
            rate_segments.append(f"{float(epcs_per_minute):.2f} EPCs/min")
        if rate_segments:
            prefix = label_parts[0] if label_parts else "Indicador de modo"
            return f"{prefix} — {', '.join(rate_segments)}"
        if label_parts:
            return label_parts[0]
    return None


class _ExecutiveCollector:
    """Internal helper assembling canonical executive KPI entries."""

    def __init__(
        self,
        summary_df: pd.DataFrame | None,
        *,
        structured_info: dict | None = None,
        continuous_info: dict | None = None,
        logistics_info: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.summary_df = summary_df if summary_df is not None else pd.DataFrame()
        self.structured_info = structured_info or {}
        self.continuous_info = continuous_info or {}
        self.logistics_info = logistics_info or {}
        self.metadata = metadata or {}
        self.entries: list[ExecutiveEntry] = []

    def _append_executive(
        self,
        indicator: str,
        result: str | None,
        interpretation: str | None,
    ) -> None:
        """Append a canonical KPI entry ensuring interpretation is present."""

        if indicator not in CANONICAL_KPI_SET:
            raise ValueError(f"Unknown executive KPI name: {indicator}")
        result_str = result if result else "N/A"
        interpretation_str = (
            interpretation
            if interpretation and interpretation.strip()
            else "Sem interpretação disponível."
        )
        self.entries.append(
            ExecutiveEntry(
                indicator=indicator,
                result=result_str,
                interpretation=interpretation_str,
            )
        )

    def build(self) -> list[ExecutiveEntry]:
        """Return all canonical entries in the mandated order."""

        for indicator in CANONICAL_KPI_ORDER:
            handler = getattr(self, f"_handle_{indicator.replace(' ', '_').replace('/', '_')}")
            handler()
        return self.entries

    # Individual KPI handlers -------------------------------------------------

    def _handle_CoverageRate(self) -> None:
        info = self.structured_info
        coverage = info.get("coverage_rate")
        expected_total = info.get("expected_total")
        expected_found = info.get("expected_found")
        result = None
        if coverage is not None and not _is_missing(coverage):
            result = _format_percentage(coverage)
            if (
                expected_total is not None
                and not _is_missing(expected_total)
                and expected_found is not None
                and not _is_missing(expected_found)
                and expected_total
            ):
                result += f" ({int(expected_found)}/{int(expected_total)})"
        interpretation = _interpret_percentage(
            coverage, context="Cobertura de EPCs esperados"
        )
        self._append_executive("CoverageRate", result, interpretation)

    def _handle_TotalDistinctEPCs(self) -> None:
        total = None
        if not self.summary_df.empty:
            if "EPC" in self.summary_df.columns:
                try:
                    total = int(self.summary_df["EPC"].nunique())
                except Exception:  # pragma: no cover - defensive fallback
                    total = int(self.summary_df.shape[0])
            else:
                total = int(self.summary_df.shape[0])
        result = str(total) if total is not None else None
        interpretation = (
            f"{total} EPCs distintos processados na sessão."
            if total is not None
            else "Nenhum EPC distinto identificado."
        )
        self._append_executive("TotalDistinctEPCs", result, interpretation)

    def _handle_Total_de_Cajas_Leydo(self) -> None:
        value = self.logistics_info.get("total_logistics_epcs")
        result = str(int(value)) if value is not None and not _is_missing(value) else None
        interpretation = (
            f"{int(value)} totes confirmados nas janelas logísticas."
            if value is not None and not _is_missing(value)
            else "Sem totes logísticos processados."
        )
        self._append_executive("Total de Cajas Leydo", result, interpretation)

    def _handle_Tasa_promedio_de_lectura_por_intento(self) -> None:
        value = self.logistics_info.get("attempt_success_rate_pct")
        result = _format_percentage(value) if value is not None and not _is_missing(value) else None
        interpretation = _interpret_percentage(
            value, context="Taxa média de leitura por intento"
        )
        self._append_executive(
            "Tasa promedio de lectura por intento", result, interpretation
        )

    def _handle_Tiempo_promedio_de_leitura_por_tote(self) -> None:
        value = self.logistics_info.get("tote_cycle_time_seconds")
        result = _format_seconds(value) if value is not None and not _is_missing(value) else None
        if value is None or _is_missing(value):
            interpretation = "Sem medições de ciclo por tote."
        else:
            numeric = float(value)
            if numeric <= 45:
                qualifier = "fluxo rápido"
            elif numeric <= 90:
                qualifier = "fluxo moderado"
            else:
                qualifier = "fluxo lento"
            interpretation = f"Tempo médio por tote indica {qualifier} ({numeric:.2f} s)."
        self._append_executive(
            "Tiempo promedio de leitura por tote", result, interpretation
        )

    def _handle_Tasa_de_fallas_de_leitura(self) -> None:
        value = self.logistics_info.get("attempt_failure_rate_pct")
        result = _format_percentage(value) if value is not None and not _is_missing(value) else None
        interpretation = _interpret_percentage(
            value, context="Taxa de falhas de leitura", high_is_good=False
        )
        self._append_executive(
            "Tasa de fallas de leitura", result, interpretation
        )

    def _handle_Tasa_de_lecturas_duplicadas(self) -> None:
        value = self.logistics_info.get("duplicate_reads_per_tote")
        result = _format_ratio(value) if value is not None and not _is_missing(value) else None
        if value is None or _is_missing(value):
            interpretation = "Sem dados de leituras duplicadas."
        else:
            numeric = float(value)
            if numeric <= 1.5:
                qualifier = "controle adequado"
            elif numeric <= 3.0:
                qualifier = "atenção moderada"
            else:
                qualifier = "alto volume de duplicadas"
            interpretation = (
                f"Média de {numeric:.2f} leituras duplicadas por tote, {qualifier}."
            )
        self._append_executive(
            "Tasa de lecturas duplicadas", result, interpretation
        )

    def _handle_Cobertura_del_área_de_leitura(self) -> None:
        value = self.logistics_info.get("coverage_pct")
        result = _format_percentage(value) if value is not None and not _is_missing(value) else None
        interpretation = _interpret_percentage(
            value, context="Cobertura da área de leitura"
        )
        self._append_executive(
            "Cobertura del área de leitura", result, interpretation
        )

    def _handle_Capacidad_de_lectura_simultánea(self) -> None:
        value = self.logistics_info.get("concurrent_capacity")
        avg_value = self.logistics_info.get("concurrent_capacity_avg")
        peak_time = self.logistics_info.get("concurrent_capacity_time")
        result = None
        if value is not None and not _is_missing(value):
            result = f"{int(value)} totes"
            timestamp = _format_timestamp(peak_time)
            if timestamp:
                result += f" @ {timestamp}"
        if value is None or _is_missing(value):
            interpretation = "Sem dados de capacidade simultânea."
        else:
            if avg_value is None or _is_missing(avg_value):
                interpretation = f"Pico de {int(value)} totes em paralelo."
            else:
                interpretation = (
                    f"Capacidade simultânea média de {float(avg_value):.2f} com pico de {int(value)}."
                )
        self._append_executive(
            "Capacidad de lectura simultánea", result, interpretation
        )

    def _handle_Disponibilidad_del_sistema(self) -> None:
        value = self.logistics_info.get("reader_uptime_pct")
        uptime_seconds = self.logistics_info.get("reader_uptime_seconds")
        scheduled_seconds = self.logistics_info.get("scheduled_session_seconds")
        result = _format_percentage(value) if value is not None and not _is_missing(value) else None
        interpretation = _interpret_percentage(
            value, context="Disponibilidade do sistema"
        )
        if (
            uptime_seconds is not None
            and not _is_missing(uptime_seconds)
            and scheduled_seconds is not None
            and not _is_missing(scheduled_seconds)
            and scheduled_seconds
        ):
            interpretation += (
                f" ({float(uptime_seconds):.0f}s ativos de {float(scheduled_seconds):.0f}s agendados)."
            )
        self._append_executive("Disponibilidad del sistema", result, interpretation)

    def _handle_AverageRSSI(self) -> None:
        structured = self.structured_info.get("global_rssi_avg")
        continuous = self.continuous_info.get("global_rssi_avg")
        value = structured if structured is not None and not _is_missing(structured) else continuous
        result = f"{float(value):.2f} dBm" if value is not None and not _is_missing(value) else None
        interpretation = _interpret_rssi(value)
        self._append_executive("AverageRSSI", result, interpretation)

    def _handle_RSSI_StdDev(self) -> None:
        structured = self.structured_info.get("global_rssi_std")
        continuous = self.continuous_info.get("global_rssi_std")
        value = structured if structured is not None and not _is_missing(structured) else continuous
        result = f"{float(value):.2f} dBm" if value is not None and not _is_missing(value) else None
        interpretation = _interpret_variability(value, context="Variação global de RSSI")
        self._append_executive("RSSI_StdDev", result, interpretation)

    def _handle_AntennaBalance(self) -> None:
        value = self.structured_info.get("antenna_balance")
        result = _format_percentage(value) if value is not None and not _is_missing(value) else None
        interpretation = _interpret_balance(value)
        self._append_executive("AntennaBalance", result, interpretation)

    def _handle_TagReadRedundancy(self) -> None:
        value = self.structured_info.get("tag_read_redundancy")
        result = _format_ratio(value) if value is not None and not _is_missing(value) else None
        interpretation = _interpret_redundancy(value)
        self._append_executive("TagReadRedundancy", result, interpretation)

    def _handle_TagDwellTimeAvg(self) -> None:
        value = self.continuous_info.get("average_dwell_seconds")
        result = f"{float(value):.2f} s" if value is not None and not _is_missing(value) else None
        interpretation = _interpret_dwell(value)
        self._append_executive("TagDwellTimeAvg", result, interpretation)

    def _handle_ConcurrentTagsPeak(self) -> None:
        value = self.continuous_info.get("concurrency_peak")
        average = self.continuous_info.get("concurrency_average")
        timestamp = _format_timestamp(self.continuous_info.get("concurrency_peak_time"))
        result = None
        if value is not None and not _is_missing(value):
            result = str(int(value))
            if timestamp:
                result += f" @ {timestamp}"
        interpretation = _interpret_concurrency(value, average)
        self._append_executive("ConcurrentTagsPeak", result, interpretation)

    def _handle_ReadContinuityRate(self) -> None:
        value = self.continuous_info.get("read_continuity_rate")
        result = _format_percentage(value) if value is not None and not _is_missing(value) else None
        interpretation = _interpret_percentage(
            value, context="Taxa de continuidade de leitura"
        )
        self._append_executive("ReadContinuityRate", result, interpretation)

    def _handle_ThroughputPerMinute(self) -> None:
        value = self.continuous_info.get("throughput_per_minute")
        result = f"{float(value):.2f} EPCs/min" if value is not None and not _is_missing(value) else None
        interpretation = _interpret_throughput(value)
        self._append_executive("ThroughputPerMinute", result, interpretation)

    def _handle_SessionDuration(self) -> None:
        value = self.continuous_info.get("session_duration_seconds")
        result = _format_seconds(value) if value is not None and not _is_missing(value) else None
        if value is None or _is_missing(value):
            interpretation = "Sem duração consolidada da sessão."
        else:
            active = self.continuous_info.get("session_active_seconds")
            active_text = ""
            if active is not None and not _is_missing(active):
                active_text = f" (tempo ativo {float(active):.2f} s)"
            interpretation = (
                f"Sessão monitorada por {float(value):.2f} s{active_text}."
            )
        self._append_executive("SessionDuration", result, interpretation)

    def _handle_ModePerformance(self) -> None:
        structured_mode = format_mode_indicator(self.structured_info)
        continuous_mode = format_mode_indicator(self.continuous_info)
        combined = continuous_mode or structured_mode
        result = combined if combined else None
        interpretation = (
            combined
            if combined
            else "Sem descrição consolidada do desempenho do modo."
        )
        self._append_executive("ModePerformance", result, interpretation)

    def _handle_NoiseIndicator(self) -> None:
        structured_indicator = self.structured_info.get("rssi_noise_indicator")
        continuous_indicator = self.continuous_info.get("rssi_noise_indicator")
        indicator = (
            continuous_indicator
            if continuous_indicator
            else structured_indicator
        )
        structured_flag = self.structured_info.get("rssi_noise_flag")
        continuous_flag = self.continuous_info.get("rssi_noise_flag")
        flag = (
            continuous_flag
            if continuous_flag is not None and not _is_missing(continuous_flag)
            else structured_flag
        )
        noise_reads = self.continuous_info.get("rssi_noise_reads_per_epc")
        if noise_reads is None or _is_missing(noise_reads):
            noise_reads = self.structured_info.get("rssi_noise_reads_per_epc")
        interpretation = _interpret_noise(
            indicator,
            noise_flag=bool(flag) if flag is not None and not _is_missing(flag) else None,
            noise_reads=noise_reads if noise_reads is not None and not _is_missing(noise_reads) else None,
        )
        result = indicator if indicator else None
        self._append_executive("NoiseIndicator", result, interpretation)


def build_executive_kpi_table(
    summary_df: pd.DataFrame | None,
    *,
    structured_info: dict | None = None,
    continuous_info: dict | None = None,
    logistics_info: dict | None = None,
    metadata: dict | None = None,
) -> pd.DataFrame:
    """Return a canonical executive KPI table with mandated headers."""

    collector = _ExecutiveCollector(
        summary_df,
        structured_info=structured_info,
        continuous_info=continuous_info,
        logistics_info=logistics_info,
        metadata=metadata,
    )
    entries = collector.build()
    rows = [entry.as_row() for entry in entries]
    return pd.DataFrame(rows, columns=["Indicador", "Resultado", "Interpretação executiva"])


def iter_executive_kpis(
    summary_df: pd.DataFrame | None,
    *,
    structured_info: dict | None = None,
    continuous_info: dict | None = None,
    logistics_info: dict | None = None,
    metadata: dict | None = None,
) -> Iterable[ExecutiveEntry]:
    """Yield canonical KPI entries preserving order."""

    collector = _ExecutiveCollector(
        summary_df,
        structured_info=structured_info,
        continuous_info=continuous_info,
        logistics_info=logistics_info,
        metadata=metadata,
    )
    yield from collector.build()
