import sqlite3
import asyncio
from typing import List, Any, Set, Tuple

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ExecutionAccuracyMetric(BaseMetric):
    """
    Métrica para avaliar a precisão da execução de consultas SQL.
    Compara o resultado de uma consulta gerada (actual_output) com o
    resultado de uma consulta de referência (expected_output) em um
    banco de dados SQLite (padrão Spider). A comparação é
    insensível à ordem das linhas.
    """

    def __init__(self, db_path: str, threshold: float = 1.0):
        """
        Inicializa a métrica.

        Args:
            db_path (str): O caminho para o arquivo do banco de dados SQLite.
            threshold (float): O limiar para o sucesso da métrica. O padrão é 1.0,
                               o que significa que apenas uma correspondência exata é bem-sucedida.
        """
        if not db_path:
            raise ValueError("O caminho do banco de dados (db_path) não pode ser nulo ou vazio.")
        self.db_path = db_path
        super().__init__(threshold=threshold)
        self.reason = "" 

    def _execute_query(self, cursor: sqlite3.Cursor, query: str) -> List[Tuple] | None:
        """
        Executa uma consulta SQL de forma segura e retorna os resultados.
        Retorna None se ocorrer um erro de sintaxe ou execução.
        """
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except sqlite3.Error as e:
            self.reason = f"Falha na execução da consulta: {e}. Consulta: '{query}'"
            return None

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Mede a precisão da execução da consulta SQL.

        Args:
            test_case (LLMTestCase): O caso de teste contendo as consultas
                                     'actual_output' e 'expected_output'.

        Returns:
            float: 1.0 se os resultados forem idênticos (insensível à ordem),
                   0.0 caso contrário.
        """
        if not test_case.actual_output or not test_case.expected_output:
            self.score = 0.0
            self.reason = "O 'actual_output' ou 'expected_output' está ausente no caso de teste."
            return self.score

        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            
            actual_results = self._execute_query(cursor, test_case.actual_output)
            expected_results = self._execute_query(cursor, test_case.expected_output)

            if actual_results is None or expected_results is None:
                self.score = 0.0
                return self.score

            # d. Comparar os conjuntos de resultados de forma insensível à ordem.
            # Convertemos a lista de tuplas em um conjunto de tuplas.
            # Conjuntos são inerentemente desordenados e a comparação verifica
            # se todos os elementos são os mesmos.
            actual_set = set(actual_results)
            expected_set = set(expected_results)

            if actual_set == expected_set:
                self.score = 1.0
                self.reason = "Os resultados da execução corresponderam perfeitamente."
            else:
                self.score = 0.0
                self.reason = (f"Os resultados da execução não corresponderam. "
                             f"Obtido: {len(actual_set)} linhas, Esperado: {len(expected_set)} linhas.")

        except sqlite3.Error as e:
            self.score = 0.0
            self.reason = f"Falha na conexão com o banco de dados: {e}"
        finally:
            if conn:
                conn.close()

        return self.score

    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:
        """Versão assíncrona do método measure."""
        return await asyncio.to_thread(self.measure, test_case)

    def is_successful(self) -> bool:
        """Verifica se a métrica foi bem-sucedida com base no limiar."""
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"