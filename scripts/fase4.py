import torch
import json
from peft import PeftModel
from tqdm import tqdm
import collections

# 4.1: Metodologia de Avaliação MMLU - Prompt de 4-shot fixo
MMLU_4SHOT_PROMPT = """The following are multiple choice questions (with answers) about a variety of topics.

Question: What is the capital of France?
(A) Berlin
(B) Madrid
(C) Paris
(D) Rome
Answer: C

Question: Which element has the atomic number 1?
(A) Helium
(B) Oxygen
(C) Hydrogen
(D) Lithium
Answer: C

Question: In what year did the Titanic sink?
(A) 1905
(B) 1912
(C) 1918
(D) 1923
Answer: B

Question: Who wrote the play "Hamlet"?
(A) Charles Dickens
(B) William Shakespeare
(C) Jane Austen
(D) Mark Twain
Answer: B

"""

def _formatar_pergunta_mmlu(item):
    """Formata uma única pergunta e suas opções para o prompt."""
    question = item['question']
    options = "".join([f"({k}) {v}\n" for k, v in item['options'].items()])
    return f"Question: {question}\n{options}Answer:"

@torch.no_grad()
def _avaliar_modelo_no_mmlu(model, tokenizer, mmlu_dataset):
    """
    Função 'worker' que avalia um único modelo (base ou fine-tuned) no dataset MMLU.
    Utiliza o método de log-likelihood para determinar a resposta do modelo.
    """
    results = collections.defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for item in tqdm(mmlu_dataset, desc="Avaliando no MMLU"):
        category = item['category']
        correct_answer_char = item['answer']
        
        # Formata a pergunta e cria o prompt completo
        prompt_inicio = MMLU_4SHOT_PROMPT + _formatar_pergunta_mmlu(item)
        
        log_likelihoods = []
        # 4.1: Avalia a probabilidade de cada opção de múltipla escolha
        for choice_char, choice_text in item['options'].items():
            prompt_completo = f"{prompt_inicio} {choice_char}"
            
            inputs = tokenizer(prompt_completo, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs.input_ids)
            
            # Pega o log-likelihood da última token (a resposta)
            log_likelihood = -outputs.loss.item()
            log_likelihoods.append(log_likelihood)
        
        # A resposta do modelo é a que tem maior log-likelihood
        model_answer_char = ["A", "B", "C", "D"][log_likelihoods.index(max(log_likelihoods))]
        
        # 4.2: Cálculo de Acurácia
        is_correct = (model_answer_char == correct_answer_char)
        
        results[category]['correct'] += 1 if is_correct else 0
        results[category]['total'] += 1
        results['overall']['correct'] += 1 if is_correct else 0
        results['overall']['total'] += 1

    # Calcula acurácia final
    accuracies = {}
    for cat, data in results.items():
        accuracies[cat] = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
        
    return accuracies


def executar_analise_regressao_fase4(base_model, tokenizer, mmlu_dataset_path, caminhos_modelos_ft: list):
    """
    Ponto de entrada para a Fase 4. Orquestra a avaliação de todos os modelos
    e gera o relatório de análise de regressão de capacidade.
    """
    print("--- INICIANDO FASE 4: Análise Quantitativa de Regressão de Capacidade ---")

    try:
        with open(mmlu_dataset_path, 'r', encoding='utf-8') as f:
            mmlu_dataset = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: Dataset MMLU não encontrado em '{mmlu_dataset_path}'. Verifique o caminho no .env")
        return

    # 1. Avalia o modelo base
    print("\n[Avaliando Modelo Base no MMLU...]")
    base_model.eval() # Coloca o modelo em modo de avaliação
    resultados_base = _avaliar_modelo_no_mmlu(base_model, tokenizer, mmlu_dataset)
    
    all_results = {"Modelo Base": resultados_base}

    # 2. Avalia cada modelo fine-tuned
    for path_adapter in caminhos_modelos_ft:
        print(f"\n[Carregando e Avaliando Modelo Fine-Tuned de '{path_adapter}'...]")
        try:
            # Carrega os pesos do LoRA sobre o modelo base
            ft_model = PeftModel.from_pretrained(base_model, path_adapter)
            ft_model.eval() # Modo de avaliação
            
            resultados_ft = _avaliar_modelo_no_mmlu(ft_model, tokenizer, mmlu_dataset)
            all_results[path_adapter] = resultados_ft
            
            # Limpa a memória descarregando os adaptadores (importante)
            del ft_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERRO ao carregar ou avaliar o adaptador de '{path_adapter}': {e}")

    # 4.3: Análise de Regressão
    print("\n\n--- RELATÓRIO DE ANÁLISE DE REGRESSÃO DE CAPACIDADE (MMLU) ---")
    print("-" * 60)
    
    # Imprime os resultados brutos de acurácia
    print("\nResultados de Acurácia Bruta (%):")
    for model_name, results in all_results.items():
        print(f"\nModelo: {model_name}")
        for category, acc in results.items():
            print(f"  - Categoria {category.upper()}: {acc:.2f}%")
            
    print("-" * 60)
    
    # Calcula e imprime a variação percentual
    print("\nVariação Percentual de Acurácia (Fine-Tuned vs. Base):")
    for model_name, ft_results in all_results.items():
        if model_name == "Modelo Base":
            continue
        
        print(f"\nComparando com: {model_name}")
        for category, ft_acc in ft_results.items():
            base_acc = resultados_base.get(category, 0)
            if base_acc > 0:
                variacao = ((ft_acc - base_acc) / base_acc) * 100
                simbolo = "↑" if variacao > 0 else "↓"
                print(f"  - Categoria {category.upper()}: {variacao:+.2f}% {simbolo}")
            else:
                print(f"  - Categoria {category.upper()}: N/A (Acurácia base era 0)")
    
    print("\n--- FASE 4 CONCLUÍDA ---")