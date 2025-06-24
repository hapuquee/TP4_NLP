# scripts/infer_sql.py

import sqlite3
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def extrair_sql(model_output: str) -> str:
    """Extrai SQL de saídas geradas pelo modelo."""
    padrao_bloco_codigo = re.compile(r"```(?:sql)?\n?(.*?)```", re.DOTALL)
    match = padrao_bloco_codigo.search(model_output)
    if match:
        return match.group(1).strip()

    padrao_aspas = re.compile(r'"(SELECT .*?)"', re.IGNORECASE)
    match = padrao_aspas.search(model_output)
    if match:
        return match.group(1).strip()

    if model_output.strip().upper().startswith('SELECT'):
        return model_output.strip()

    return ""

def gerar_sqls_e_salvar_no_banco(model, tokenizer, db_path="tp4.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, question FROM test_case")
    perguntas = cursor.fetchall()

    for id_, question in perguntas:
        conversation = [
            {"role": "system", "content": prompt_fewshot},
            {"role": "user", "content": question}
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids_length = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        generated_tokens = outputs[0][input_ids_length:]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        sql_resultado = extrair_sql(decoded_output)

        # Atualiza o campo actual_result na tabela
        cursor.execute(
            "UPDATE test_case SET actual_result = ? WHERE id = ?",
            (sql_resultado, id_)
        )
        print(f"Pergunta ID {id_}: SQL gerado -> {sql_resultado}")

    conn.commit()
    conn.close()
    print("Todos os resultados foram salvos na tabela.")