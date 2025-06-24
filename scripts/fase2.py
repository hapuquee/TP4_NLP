# fase2.py
import os
import torch
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

def create_dataset(file_path):
    """
    Loads data from a JSON file and formats it into a question-answer dataset.
    Each item in the dataset will have a 'text' field formatted as:
    '<s>[INST] {question} [/INST] {answer} </s>'
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        question = item['question']
        answer = item['query']
        # This specific format is important for instruction-tuned models like Mistral
        formatted_data.append(f"<s>[INST] {question} [/INST] {answer} </s>")

    return Dataset.from_dict({'text': formatted_data})

def _executar_um_treino(
    base_model,
    tokenizer,
    train_dataset,
    lora_config_params: dict,
    training_config_params: dict,
    output_dir_suffix: str
):
    print(f"--- INICIANDO EXPERIMENTO DE FINE-TUNING: {output_dir_suffix} ---")

    print("\n[CONFIGURAÇÃO LORA]")
    for key, value in lora_config_params.items():
        print(f"  - {key}: {value}")
    lora_config = LoraConfig(**lora_config_params)
    
    model_lora = get_peft_model(base_model, lora_config)
    model_lora.print_trainable_parameters()

    output_dir = f"./results/{output_dir_suffix}"
    training_config_params['output_dir'] = output_dir

    print("\n[CONFIGURAÇÃO DE TREINAMENTO]")
    for key, value in training_config_params.items():
        print(f"  - {key}: {value}")
    
    training_args = SFTConfig(**training_config_params)

    trainer = SFTTrainer(
        model=model_lora,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
        tokenizer=tokenizer,
    )

    torch.cuda.empty_cache()
    
    print("\n[INICIANDO TREINAMENTO QLORA...]")
    os.environ["WANDB_DISABLED"] = "true"
    try:
        trainer.train()
        print("\nTreinamento concluído com sucesso!")
    except Exception as e:
        print(f"\nERRO DURANTE O TREINAMENTO: {e}")
    
    final_output_dir = f"./{output_dir_suffix}_final_model_adapter"
    trainer.save_model(final_output_dir)
    print(f"\nAdaptador LoRA salvo em: {final_output_dir}")
    print(f"--- EXPERIMENTO {output_dir_suffix} CONCLUÍDO ---\n")

    return final_output_dir

def executar_todos_experimentos_fase2(base_model, tokenizer, train_dataset):
    """
    Ponto de entrada para a Fase 2.
    Define e orquestra a execução de múltiplos experimentos de fine-tuning.
    """
    print("--- INICIANDO FASE 2: Execução do Fine-Tuning ---")
    
    # 2.2: Configuração e Documentação dos parâmetros LoRA (base para todos os experimentos)
    lora_params = {
        'r': 8,
        'lora_alpha': 16,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'lora_dropout': 0.05,
        'bias': "none",
        'task_type': "CAUSAL_LM",
    }
    
    # Parâmetros de treino que serão comuns a todos os experimentos
    base_training_params = {
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 4,
        'optim': "adamw_torch",
        'save_steps': 0, # Salvaremos apenas no final de cada treino
        'logging_steps': 10,
        'weight_decay': 0.001,
        'fp16': False,
        'bf16': True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        'max_grad_norm': 0.3,
        'max_steps': -1,
        'warmup_ratio': 0.03,
        'group_by_length': True,
        'lr_scheduler_type': "cosine",
        'dataset_text_field': "text",
        'report_to': "none"
    }

    # 2.3: Experimentação de Hiperparâmetros
    # Experimento 1: Taxa de aprendizado mais alta (1e-4) e menos épocas (3)
    exp1_training_params = base_training_params.copy()
    exp1_training_params.update({
        'learning_rate': 1e-4,
        'num_train_epochs': 3,
    })

    # Experimento 2: Taxa de aprendizado mais baixa (5e-5) e mais épocas (5)
    exp2_training_params = base_training_params.copy()
    exp2_training_params.update({
        'learning_rate': 5e-5,
        'num_train_epochs': 5,
    })

    # Executa os experimentos em sequência
    _executar_um_treino(
        base_model=base_model, tokenizer=tokenizer, train_dataset=train_dataset,
        lora_config_params=lora_params, training_config_params=exp1_training_params,
        output_dir_suffix="exp1_lr-1e-4_epochs-3"
    )

    _executar_um_treino(
        base_model=base_model, tokenizer=tokenizer, train_dataset=train_dataset,
        lora_config_params=lora_params, training_config_params=exp2_training_params,
        output_dir_suffix="exp2_lr-5e-5_epochs-5"
    )
    
    print("--- FASE 2 CONCLUÍDA ---")