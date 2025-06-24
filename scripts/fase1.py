# fase1.py
import re

prompt_fewshot = """
Sua tarefa é transformar um texto em uma consulta SQL.

**Exemplos:**

**Texto:** "List the name of clubs in ascending alphabetical order."
**SQL:** "SELECT Name FROM club ORDER BY Name ASC"

**Texto:** "What are the names of authors who have exactly 1 paper?"
**SQL:** "SELECT T1.name FROM Author AS T1 JOIN Author_list AS T2 ON T1.author_id  =  T2.author_id GROUP BY T1.author_id HAVING count(*)  =  1"

**Texto:** "Which customers did not make any orders? List the first name, middle initial and last name."
**SQL:** "SELECT customer_first_name ,  customer_middle_initial ,  customer_last_name FROM Customers EXCEPT SELECT T1.customer_first_n…ustomer_middle_initial ,  T1.customer_last_name FROM Customers AS T1 JOIN Orders AS T2 ON T1.customer_id  =  T2.customer_id"

"""

def extrair_sql(texto_do_modelo):
    """
    Extrai uma consulta SQL do texto de saída de um modelo.
    Tenta encontrar um bloco de código (```) primeiro, depois uma string entre aspas.
    """
    padrao_bloco_codigo = re.compile(r"```(?:sql)?\n?(.*?)```", re.DOTALL)
    match = padrao_bloco_codigo.search(texto_do_modelo)
    if match:
        return match.group(1).strip()

    padrao_aspas = re.compile(r'"(SELECT .*?)"', re.IGNORECASE)
    match = padrao_aspas.search(texto_do_modelo)
    if match:
        return match.group(1).strip()

    if texto_do_modelo.strip().upper().startswith('SELECT'):
        return texto_do_modelo.strip()

    return ""

def executar_baseline_fase1(model, tokenizer, task_avaliation_dataset):
    """
    Executa a avaliação de baseline (Fase 1), submetendo o modelo ao dataset
    de avaliação e comparando os resultados.

    Args:
        model: O modelo de linguagem pré-treinado.
        tokenizer: O tokenizador associado ao modelo.
        task_avaliation_dataset: O dataset de avaliação (ex: Spider dev split).
    
    Returns:
        Uma tupla contendo (contagem_de_sucessos, total_de_itens_avaliados).
    """
    print("--- INICIANDO FASE 1: Estabelecimento do Baseline de Desempenho ---")
    
    resultados = []

    for item in task_avaliation_dataset[:5]:
        conversation = [{"role": "system",
                        "content": f'{prompt_fewshot}'},
                        {
                            "role": "user",
                            "content": item['question']
                        }]

        inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
        )

        input_ids_length = inputs['input_ids'].shape[1]
        outputs = model.generate(**inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id)
        generated_tokens = outputs[0][input_ids_length:]
        model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        sql_model = extrair_sql(model_output)

        item_resultado = {
            'pergunta': item['question'],
            'sql_do_modelo': sql_model,
            'sql_esperado': item['query']
        }
        resultados.append(item_resultado)
        print(f"Entrada: {item['question']}")
        print(f"Saída do Modelo:\n{tokenizer.decode(generated_tokens, skip_special_tokens=True)}")
        print(f"Saída Esperada: {item['query']}")
        print("-" * 50)
        
    count = 0
    for i in range(len(resultados)):
        modelo = (resultados[i]['sql_do_modelo']).lower()
        expected = (resultados[i]['sql_esperado']).lower()
        print(f"Modelo: {modelo}")
        print(f"Esperado: {expected}")
        print("-"*50)
        if modelo == expected:
            count += 1

    print(f"Taxa de acerto: {count*100}%")

    print("\n--- FASE 1 CONCLUÍDA ---")
    
    return count, len(resultados)