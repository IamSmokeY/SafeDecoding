from utils.opt_utils import load_model_and_tokenizer
from peft import PeftModel
from utils.safe_decoding import SafeDecoding
from utils.string_utils import PromptManager, load_conversation_template
import torch

class LLM:
    def __init__(self, 
                 model_name="llama2",
                 device="cuda:0", 
                 FP16=True, 
                 low_cpu_mem_usage=True, 
                 use_cache=True, 
                 do_sample=False):
        
        if model_name == "llama2":
            self.model_name = "meta-llama/Llama-2-7b-chat-hf"
            self.template_name = 'llama-2'
        else:
            raise ValueError("Only 'llama2' is supported as the model name.")

        model, self.tokenizer = load_model_and_tokenizer(self.model_name, 
                       FP16=FP16,
                       low_cpu_mem_usage=low_cpu_mem_usage,
                       use_cache=use_cache,
                       do_sample=do_sample,
                       device=device) 
        self.model = PeftModel.from_pretrained(model, "../lora_modules/"+model_name, adapter_name="expert")
        self.conv_template = load_conversation_template(self.template_name)
        # Initialize contrastive decoder
        self.safe_decoder = SafeDecoding(model, 
                                          self.tokenizer, 
                                          ['base', 'expert'], 
                                          alpha=3, 
                                          first_m=2, 
                                          top_k=10, 
                                          num_common_tokens=5,
                                          verbose=True)
    def generate(self, prompt):
        input_manager = PromptManager(tokenizer=self.tokenizer, 
            conv_template=self.conv_template, 
            instruction=prompt,
            whitebox_attacker=True)
        inputs = input_manager.get_inputs()
        outputs, output_length = self.safe_decoder.safedecoding_lora(inputs, None)
        return outputs
