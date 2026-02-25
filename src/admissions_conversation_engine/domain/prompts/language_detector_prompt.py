from __future__ import annotations

LANGUAGE_DETECTOR_PROMPT = """
Detecta el idioma principal del mensaje del usuario.
Responde ÚNICAMENTE un JSON válido (sin markdown) con:
- language: string (elige SOLO entre: {allowed_languages})
- confidence: number (0 a 1)

Reglas:
- Si no puedes inferir con confianza, usa "{language_fallback}".
- Sin texto adicional.

Mensaje usuario:
{{user_message}}
"""
