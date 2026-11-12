"""
Módulo de observabilidad y monitoreo continuo para el proyecto MiBici.
Implementa detección de Data Drift y Data Quality utilizando Evidently.
"""

from mibici.monitoring.drift import (
    preparar_conjuntos_drift,
    ejecutar_reporte_drift,
)

__all__ = ["preparar_conjuntos_drift", "ejecutar_reporte_drift"]
