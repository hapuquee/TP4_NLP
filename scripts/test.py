import sqlite3
import pytest
from custom_metrics.accuracy import ExecutionAccuracyMetric
from deepeval.test_case import LLMTestCase

DB_PATH = "tp4.db"
TABLE_NAME = "test_case"

@pytest.fixture(scope="module")
def accuracy_metric():
    """Inicializa a métrica uma vez por módulo."""
    return ExecutionAccuracyMetric(db_path=DB_PATH)

@pytest.fixture
def carregar_casos_de_teste():
    """Lê os dados do banco e retorna os test cases."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"SELECT question, expected_result, actual_result FROM {TABLE_NAME}")
    rows = cursor.fetchall()
    conn.close()

    test_cases = []
    for question, expected_result, actual_result in rows:
        if expected_result and actual_result:
            test_cases.append(
                LLMTestCase(
                    input=question,
                    expected_output=expected_result,
                    actual_output=actual_result,
                )
            )
    return test_cases

def test_accuracy_per_query(accuracy_metric, carregar_casos_de_teste):
    """Testa a acurácia de cada consulta individualmente."""
    for case in carregar_casos_de_teste:
        score = accuracy_metric.measure(case)
        print(f"Query: {case.input[:50]}... -> Score: {score}, Motivo: {accuracy_metric.reason}")
        assert score in [0.0, 1.0], "Score inválido"

def test_overall_accuracy(accuracy_metric, carregar_casos_de_teste):
    """Calcula a acurácia total do modelo."""
    total = len(carregar_casos_de_teste)
    acertos = 0

    for case in carregar_casos_de_teste:
        score = accuracy_metric.measure(case)
        acertos += score

    acc = acertos / total if total else 0
    print(f"Acurácia total: {acc:.2%}")
    assert acc >= 0.0  
