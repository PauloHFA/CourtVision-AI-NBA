"""Protocolo de avaliação do assistente.

O conjunto curado de 80 perguntas para AC-08 ainda não existe.
`rag_protocol` mede só o fixture sintético e não aceita o gate.
"""

from courtvision.evaluation.rag_protocol import evaluate_assistant, protocol_summary

__all__ = ["evaluate_assistant", "protocol_summary"]
