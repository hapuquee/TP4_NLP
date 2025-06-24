# Avaliação de Modelo Text-to-SQL com Fine-Tuning
Este projeto tem como objetivo avaliar o desempenho do modelo de linguagem Mistral-7B-Instruct-v0.3 em tarefas de conversão de texto para SQL (Text-to-SQL). O processo inclui uma avaliação inicial (baseline), fine-tuning com diferentes hiperparâmetros e uma avaliação final dos modelos ajustados.

Estrutura do Projeto
O script main.py orquestra todo o processo, que é dividido nas seguintes fases:

Carregamento de Recursos: Carrega o modelo base da Hugging Face, os datasets de fine-tuning, de avaliação da tarefa e o tokenizador.
Fase 1: Avaliação Baseline: Executa uma avaliação inicial do modelo pré-treinado na tarefa Text-to-SQL para estabelecer uma métrica de base.
Fase 2: Fine-Tuning: Realiza múltiplos experimentos de fine-tuning no modelo base, utilizando diferentes configurações de hiperparâmetros (como taxa de aprendizado e número de épocas).
Inferência e Salvamento: Após o fine-tuning, o melhor modelo adaptado é carregado para gerar consultas SQL a partir de um conjunto de dados de teste. As consultas geradas são então armazenadas em um banco de dados SQLite para análise posterior.
Teste Final: Um conjunto de testes pytest é utilizado para validar a performance do modelo ajustado, comparando as SQLs geradas com as esperadas.
Como Executar
Siga os passos abaixo na ordem correta para configurar e rodar o projeto.

1. Configuração do Ambiente
Antes de começar, você precisa configurar as variáveis de ambiente. Crie um arquivo chamado .env na raiz do projeto, seguindo o modelo do .env.example:

Bash

#.env.example

# Token de acesso da Hugging Face para carregar modelos privados ou com gated access
HF_TOKEN="SEU_TOKEN_AQUI"

# Caminho para o arquivo JSON do dataset de fine-tuning
PATH_FT_DATASET="caminho/para/seu/dataset_ft.json"

# Caminho para o arquivo JSON do dataset de avaliação da tarefa
PATH_TASK_EVAL_DATASET="caminho/para/seu/dataset_eval.json"

# Caminho onde o banco de dados SQLite será criado/salvo
DB_PATH="caminho/para/seu/banco.db" 
Substitua os valores de exemplo pelos caminhos e tokens corretos no seu arquivo .env.

2. Instalação das Dependências
Este projeto utiliza uma lista de bibliotecas Python que precisam ser instaladas. Use o pip para instalar todas as dependências listadas no arquivo requirements.txt:

Bash

pip install -r requirements.txt
3. Execução do Script Principal
Após configurar o ambiente e instalar as dependências, execute o script main.py. Este comando iniciará todo o processo de avaliação e fine-tuning do modelo. Este passo pode levar um tempo considerável, dependendo do hardware disponível, pois envolve o treinamento do modelo.

Bash

python main.py
4. Executando os Testes com Pytest
Após a execução bem-sucedida do main.py (que treina o modelo e salva os resultados), você pode verificar o desempenho final do modelo ajustado usando pytest. O arquivo de teste scripts/test.py foi preparado para essa validação.

Para rodar os testes, execute o seguinte comando na raiz do projeto:

Bash

pytest scripts/test.py
Isso irá iniciar os testes que carregam o modelo fine-tuned, geram as consultas SQL e as comparam com as consultas de referência para validar a precisão.
