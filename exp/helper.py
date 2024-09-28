import torch
import sys
from utils.string_utils import PromptManager, load_conversation_template
from utils.opt_utils import load_model_and_tokenizer
from utils.safe_decoding import SafeDecoding
import numpy as np
import time, logging
from peft import PeftModel


class Args:
    def __init__(self):
        # Experiment Settings
        self.model_name = "llama2"
        self.attacker = "GCG"
        self.max_new_tokens = 1024
        self.alpha = 3.0
        self.first_m = 2
        self.top_k = 10
        self.num_common_tokens = 5

        # System Settings
        self.device = "0"
        self.verbose = False
        self.FP16 = True
        self.low_cpu_mem_usage = True
        self.use_cache = False
        self.seed = 0
        self.do_sample = False
        self.top_p = None


class SafeDecodingManager:
    def __init__(self, args: Args = None):
        if args is None:
            args = Args()
        self.args = args
        self._set_random_seeds()
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.adapter_names = ['base', 'expert']
        self.model = PeftModel.from_pretrained(
            self.model, f"../lora_modules/{self.args.model_name}", adapter_name="expert"
        )
        self.whitebox_attacker = self.args.attacker in ["GCG", "AutoDAN"]
        self.safe_decoder = SafeDecoding(
            self.model,
            self.tokenizer,
            self.adapter_names,
            alpha=self.args.alpha,
            first_m=self.args.first_m,
            top_k=self.args.top_k,
            num_common_tokens=self.args.num_common_tokens,
            verbose=self.args.verbose
        )
        self.conv_template = load_conversation_template(
            'vicuna' if self.args.model_name == "vicuna" else 'llama-2'
        )

    def _set_random_seeds(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        logging.info("Random seeds are set.")

    def _load_model_and_tokenizer(self):
        if self.args.model_name == "vicuna":
            model_name = "lmsys/vicuna-7b-v1.5"
        elif self.args.model_name == "llama2":
            model_name = "meta-llama/Llama-2-7b-chat-hf"
        else:
            raise ValueError(f"Unsupported model name: {self.args.model_name}")

        logging.info(f"Loading model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            FP16=self.args.FP16,
            low_cpu_mem_usage=self.args.low_cpu_mem_usage,
            use_cache=self.args.use_cache,
            do_sample=self.args.do_sample,
            device=f'cuda:{self.args.device}'
        )
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    def generate(self, user_prompt: str) -> str:
        logging.info(f"User Prompt: \"{user_prompt}\"")
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = self.args.max_new_tokens
        gen_config.do_sample = self.args.do_sample
        gen_config.top_p = self.args.top_p

        time_start = time.time()
        input_manager = PromptManager(
            tokenizer=self.tokenizer,
            conv_template=self.conv_template,
            instruction=user_prompt,
            whitebox_attacker=self.whitebox_attacker
        )
        inputs = input_manager.get_inputs()
        outputs, output_length = self.safe_decoder.safedecoding_lora(inputs, gen_config=gen_config)
        logging.info("Generation completed.")
        return outputs