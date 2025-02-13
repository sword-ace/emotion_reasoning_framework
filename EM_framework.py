%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

# unsloth/Qwen2.5-14B-Instruct-bnb-4bit
# unsloth/Qwen2.5-32B-bnb-4bit
# unsloth/gemma-2-27b-bnb-4bit

# unsloth/gemma-7b-it-bnb-4bit

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/mistral-7b-v0.3-bnb-4bit", #"unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit", #"unsloth/Qwen2.5-14B-Instruct-bnb-4bit", #"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Mistral-Nemo-Instruct-2407",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Assuming the path to your dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import re

import pandas as pd
import json

class DataLoader:
    def __init__(self, existing_data=None):
        """
        Initialize the DataLoader with an optional existing DataFrame.
        :param existing_data: Existing DataFrame to which new data will be added.
        """
        if existing_data is None:
            self.data = pd.DataFrame()
        else:
            self.data = existing_data

    def load(self, json_input):
        """
        Load JSON data into the DataFrame, mapping 'utter' to 'ticker', 'label' to 'target',
        and using the specific parts of 'facts' as 'summary' after processing.
        :param json_input: JSON data as a string (file path) or list of dictionaries.
        """
        # If json_input is a file path, load the JSON data from the file
        if isinstance(json_input, str):
            try:
                with open(json_input, 'r') as file:
                    json_data = json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"The file at path {json_input} was not found.")
            except json.JSONDecodeError:
                raise ValueError(f"The file at path {json_input} is not a valid JSON file.")
        else:
            json_data = json_input

        # Initialize a list to collect data rows
        data_rows = []

        # Define a mapping for labels
        label_map = {
            "hap": "happy",
            "sad": "sad",
            "neu": "neutral",
            "ang": "angry",
            "exc": "excited",
            "fru": "frustrated",
        }

        # Process each entry to extract and adjust fields 'situation', 'speaker_perspective', 'impact'
        for key, entry in json_data.items():
            print(entry)

            # Create a dictionary for each row of data
            row_data = {
                'ticker': entry['utter'],  # Map 'utter' to 'ticker'
                'dialog': entry['dialog'],
                'speaker': entry['speaker'],
                'target': label_map.get(entry['label'], "Unknown label")   # Map 'label' to 'target'
            }

            data_rows.append(row_data)

        # Create a DataFrame from the collected rows
        new_data = pd.DataFrame(data_rows)

        # Concatenate this data to the existing DataFrame
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        return self.data


APPRAISAL_INSTRUCTIONS = """
Analyze the target utterance within its dialogue context, considering the provided gold label. Deliver a concise appraisal based on the following steps, and then predict an emotion label using the appraisal:

1. Highlight the key elements of the situation from the given dialogue.
2. Evaluate how the utterance aligns with the speaker's intentions and expectations.
3. Combine the analysis from steps 1 and 2 to determine whether the target utterance supports, contradicts, or remains neutral towards the speaker's intentions and expectations.

Keep the appraisal concise (1-2 sentences per step). Use the appraisal as the basis for predicting the emotion label.

Dialogue: {dialog}
Utterance: {utterance}
Gold Label: {target}

Response Format:
Appraisal: [Step-by-step concise appraisal based on the points above]
Emotion Label: [Choose one: happy, sad, neutral, angry, excited, frustrated]

Response:
"""

from datasets import Dataset

train_dataset = Dataset.from_dict(processed_data)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
############sft setting ################
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 2,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

#########this part is for emotion reasoning framework###############

class AppraisalPredictor:
    def __init__(self, ticker: str, target: str, dialog: str, model_llm=model, tokenizer=tokenizer) -> None:
        self.ticker = ticker
        self.target = target
        self.dialog = dialog
        self.prediction = ''
        self.appraisal_prompt = APPRAISAL_INSTRUCTIONS
        self.model = model_llm
        self.tokenizer = tokenizer
        self.explanation = ''
        self.__reset_model()

    def generate_appraisal(self, reset=True) -> None:
        if reset:
            self.__reset_model()

        self.scratchpad = self._generate_appraisal_prompt()
        response = self.scratchpad.split('Emotion Label:')[-1].strip()

        self.prediction = response.split()[0].strip()
        print("Prediction Result: ", self.prediction)

        self.explanation = self.scratchpad.split('Appraisal:')[-1].strip()
        self.finished = True

    def _generate_appraisal_prompt(self) -> str:
        prompt = self._construct_prompt()
        print(f"Generated Prompt: {prompt}")

        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        outputs = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=encoding.input_ids.size(1) + 300,
                num_return_sequences=1,
                temperature = 0.7,
                use_cache=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def _construct_prompt(self) -> str:
        return self.appraisal_prompt.format(
            utterance = self.ticker,
            dialog = self.dialog)

    def is_complete(self) -> bool:
        return self.finished

    def is_prediction_correct(self) -> bool:
        return check_match(self.target, self.prediction)

    def __reset_model(self) -> None:
        self.finished = False
        self.scratchpad: str = ''

def check_match(prediction, target) -> bool:
    return prediction.lower() == target.lower()


###############################do counterfactual thinking ########################
Couterfactual_thinking_instruction = """
You made wrong prediction, please perform a counterfactual analysis for the target utterance to refine your understanding of the speaker's emotional state. Follow these steps to guide your thinking:

1. Reflect on why the prediction of {previous_label} mismatches between the label and the speaker's intentions and desires based on the target utterance.
2. Imagine an alternative emotion that better aligns with the speaker's intentions and desires based on the dialog.

Keep your analysis concise and structured. Use this counterfactual analysis to propose a more accurate emotion label that fits the given context.

Dialogue: {dialog}
Utterance to analyze: {utterance}

Response Format:
Analysis: [Step-by-step concise analysis based on the points above]
Emotion Label: [Choose one: happy, sad, neutral, angry, excited, frustrated]

Response:
"""


class ReflectionModel(AppraisalPredictor):
    def __init__(self,
                 ticker: str,
                 target: str,
                 dialog: str,
                 predict_llm,
                 reflect_llm,
                 tokenizer,
                 tokenizer2) -> None:

        # Initialize the parent class for prediction capabilities
        super().__init__(ticker, target, dialog, predict_llm, tokenizer)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm
        self.reflections = []

        self.tokenizer = tokenizer2
        self.reflections_str: str = ''

        # Initialize emotion label generator for reflections
        self.emotion_label_generator = EmotionLabelGenerator(predict_llm, tokenizer)

        self.target = target
        self.dialog = dialog

        self.update_explanation = ''
        self.predict_examples = PREDICT_EXAMPLES

    def run(self, reset=True) -> None:
        # First run prediction, then potentially reflect
        super().run(reset=reset)

    def reflect(self, label) -> None:
        print('-------------Reflecting... \n')
        print(f"Emotion Label: {label}")
        print("###########################")

        # Generate updated reflection and reasoning
        model_output = self.emotion_label_generator.generate_emotion_label(utter=self.ticker, dialog=self.dialog, previous_label=label)

        self.update_explanation = model_output.split('Analysis:')[-1].strip()
        response = model_output.split('Emotion Label:')[-1]
        
        self.prediction = response.split()[0].strip()

        print(f"Correcting after reflection: {self.prediction}")
        print(f"Reasoning: {self.update_explanation}")
        print(f"True Target: {self.target}")

        reflection_result = self.is_prediction_correct()
        print(f"After reflection, correct prediction: {reflection_result}")

    def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
        # Use N-shot learning for predictions
        self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
        super().run(reset=reset)

    def is_prediction_correct(self) -> bool:
        return EMs(self.target, self.prediction)

def EMs(prediction, sentiment) -> bool:
    return prediction.strip().lower() == sentiment.strip().lower()


class EmotionLabelGenerator:
    def __init__(self, predict_llm, tokenizer):
        self.predict_llm = predict_llm
        self.tokenizer = tokenizer

    def generate_emotion_label(self, previous_label: str, utter: str, dialog: str) -> str:
        prompt = Couterfactual_thinking_instruction.format(
            previous_label=previous_label,
            dialog=dialog,
            utterance=utter)

        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        outputs = self.predict_llm.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=encoding.input_ids.size(1) + 300,
            num_return_sequences=


