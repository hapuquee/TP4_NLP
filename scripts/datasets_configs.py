import sqlite3
import json

TABLE_NAME = 'test_case'

def criar_tabela(cursor):
    """Cria a tabela no banco de dados se ela não existir."""
    # A coluna 'id' é criada como uma chave primária que se auto-incrementa.
    # As colunas 'question' e 'expected_result' não podem ser nulas (NOT NULL).
    # 'actual_result' pode ser nula (e será por padrão).
    query_criacao = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        expected_result TEXT NOT NULL,
        actual_result TEXT
    );
    """
    cursor.execute(query_criacao)
    print(f"Tabela '{TABLE_NAME}' pronta.")

def carregar_dados_do_json(caminho_arquivo):
    """Lê o arquivo JSON e retorna os dados."""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        return dados
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não é um JSON válido.")
        return None

def inserir_dados(cursor, dados):
    """Insere os dados extraídos do JSON na tabela."""
    query_insercao = f"""
    INSERT INTO {TABLE_NAME} (question, expected_result, actual_result) 
    VALUES (?, ?, ?);
    """
    
    registros_para_inserir = []
    for entrada in dados:
        question = entrada.get('question')
        expected_result = entrada.get('query')
        
        actual_result = None
        
        if question and expected_result:
            registros_para_inserir.append((question, expected_result, actual_result))
        else:
            print(f"Aviso: Ignorando entrada por falta de 'question' ou 'query': {entrada}")

    if registros_para_inserir:
        cursor.executemany(query_insercao, registros_para_inserir)
        print(f"{len(registros_para_inserir)} registros inseridos com sucesso na tabela '{TABLE_NAME}'.")

