from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
import json
from datasets import load_dataset
from scripts.datasets_configs import *
from scripts.fase1 import executar_baseline_fase1
from scripts.fase2 import executar_todos_experimentos_fase2, create_dataset
from scripts.infer_sql import gerar_sqls_e_salvar_no_banco
from scripts.fase4 import executar_analise_regressao_fase4


def load_model(token: str):
    """
    Load the Mistral-7B-Instruct-v0.3 model and tokenizer.
    This function initializes the model and tokenizer for use in inference.
    """

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")


def load_datasets(FT_DATASET_PATH,TASK_EVAL_DATASET_PATH ):

    if not FT_DATASET_PATH or not TASK_EVAL_DATASET_PATH:
        raise ValueError("Caminhos de dataset não encontrados. Verifique seu arquivo .env")

    print(f"Carregando dataset de fine-tuning de: {FT_DATASET_PATH}")
    with open(FT_DATASET_PATH, 'r', encoding='utf-8') as f:
        ft_dataset = json.load(f)
    
    print(f"Carregando dataset de avaliação de tarefa de: {TASK_EVAL_DATASET_PATH}")
    with open(TASK_EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        task_evaluation_dataset = json.load(f)

    print("Carregando dataset de avaliação geral 'MMLU'...")
    general_evaluation_dataset = load_dataset("cais/mmlu", "all")

    return ft_dataset, task_evaluation_dataset, general_evaluation_dataset

def main():
    print("Iniciando o processo de avaliação de modelo Text-to-SQL...")

    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    if not HF_TOKEN:
        print("ERRO: Token do Hugging Face (HF_TOKEN) não encontrado. Verifique seu .env")
        return
    
    FT_DATASET_PATH = os.getenv('PATH_FT_DATASET')
    TASK_EVAL_DATASET_PATH = os.getenv('PATH_TASK_EVAL_DATASET')
    DB_PATH = os.getenv('DB_PATH')


    print("Carregando modelo e tokenizador...")
    model, tokenizer = load_model(HF_TOKEN)

    print("Carregando datasets...")
    _, task_ds, general_evaluation_dataset = load_datasets(FT_DATASET_PATH,TASK_EVAL_DATASET_PATH ) 

    print("\n>>> INICIANDO FASE 1: Estabelecimento do Baseline de Desempenho<<<")
    sucessos, total_avaliado = executar_baseline_fase1(
        model=model,
        tokenizer=tokenizer,
        task_avaliation_dataset=task_ds
    )

    ft_ds = create_dataset(FT_DATASET_PATH)

    print("\n>>> INICIANDO FASE 2: Execução dos Experimentos de Fine-Tuning...")
    executar_todos_experimentos_fase2(
        base_model=model,
        tokenizer=tokenizer,
        train_dataset=ft_ds
    )

    print("\n>>> INICIANDO FASE 3: Avaliação de Desempenho na Tarefa-Alvo com Métrica Customizada<<<")
    print("Carregando os dados")
    dados_json = carregar_dados_do_json(TASK_EVAL_DATASET_PATH)

    if dados_json is None:
        return 

    try:
        conexao = sqlite3.connect(DB_PATH)
        cursor = conexao.cursor()
        
        criar_tabela(cursor)
        
        inserir_dados(cursor, dados_json)
        
        conexao.commit()
        print("Alterações salvas no banco de dados.")
        
    except sqlite3.Error as e:
        print(f"Ocorreu um erro no SQLite: {e}")
    finally:
        if 'conexao' in locals() and conexao:
            conexao.close()
            print("Conexão com o banco de dados fechada.")

    print("Fazendo pytest no modelo fine-tuning")

    model_path = "./exp2_lr-5e-5_epochs-5_final_model_adapter"  # ou o caminho que salvou
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    gerar_sqls_e_salvar_no_banco(model, tokenizer, db_path="tp4.db")

    print("\n>>> INICIANDO FASE 4: Análise de Regressão de Capacidade<<<")
    executar_analise_regressao_fase4(
        base_model=model,
        tokenizer=tokenizer,
        mmlu_dataset_path=general_evaluation_dataset,
        caminhos_modelos_ft=model_path
    )


if __name__ == '__main__':
    main()
