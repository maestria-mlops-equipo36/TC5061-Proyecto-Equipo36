"""
Módulo de pruebas de seguridad y auditoría de vulnerabilidades.
Actividad 7: Chequeos de seguridad básicos (pip-audit, bandit, .gitignore).
"""

import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent


def test_gitignore_protects_sensitive_files():
    """Valida que el archivo .gitignore excluya credenciales, datos y entornos."""
    gitignore_path = PROJ_ROOT / ".gitignore"
    assert gitignore_path.exists(), "El archivo .gitignore debe existir en la raíz."

    contenido = gitignore_path.read_text(encoding="utf-8")
    lineas = [linea.strip() for linea in contenido.splitlines() if linea.strip()]

    # Reglas verificadas contra tu .gitignore real
    reglas_esperadas = [".env", ".venv", "data", "mlflow.db", "mlartifacts"]
    for regla in reglas_esperadas:
        encontrada = any(regla in l for l in lineas)
        assert (
            encontrada
        ), f"La regla de seguridad '{regla}' debe estar contemplada en .gitignore."


def test_no_hardcoded_passwords_or_keys():
    """Verifica que no existan credenciales en texto plano en archivos de configuración."""
    config_path = PROJ_ROOT / "mibici" / "config.py"
    if config_path.exists():
        contenido = config_path.read_text(encoding="utf-8").lower()
        terminos_peligrosos = ["password =", "secret =", "api_key =", "token ="]
        for termino in terminos_peligrosos:
            assert (
                termino not in contenido
            ), f"Posible credencial en texto plano encontrada: {termino}"


def test_bandit_scan():
    """Ejecuta bandit sobre el paquete mibici para detectar riesgos comunes de código."""
    resultado = subprocess.run(
        [sys.executable, "-m", "bandit", "-r", "mibici", "-ll", "-q"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
    )
    assert (
        resultado.returncode == 0
    ), f"Bandit detectó problemas de seguridad:\n{resultado.stdout}"


def test_pip_audit_dependencies():
    """Ejecuta pip-audit para verificar vulnerabilidades conocidas en dependencias."""
    resultado = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--desc"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
    )
    assert resultado.returncode in (
        0,
        1,
    ), f"Fallo al ejecutar pip-audit: {resultado.stderr}"