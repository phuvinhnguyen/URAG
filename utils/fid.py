import logging
from typing import Any, Dict, List, Optional, Union

import torch
from haystack import Document, component
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.utils import ComponentDevice, Secret
from haystack.utils.hf import StopWordsCriteria
from transformers import AutoTokenizer, GenerationConfig, Pipeline, StoppingCriteriaList, pipeline

from utils.fid_utils import FiDConverter, FiDforConditionalGeneration

logger = logging.getLogger(__name__)


@component
class FiDGenerator(HuggingFaceLocalGenerator):
    def __init__(
        self,
        model: str = "Intel/fid_flan_t5_base_nq",
        task: str = "text2text-generation",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = {},
        stop_words: Optional[List[str]] = None,
    ):
        super(FiDGenerator, self).__init__(
            model=model,
            task=task,
            device=device,
            token=token,
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs.copy(),
            stop_words=stop_words,
        )

        # FiD models don't always have their own tokenizer, use base T5 tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception:
            # Fallback to base T5 tokenizer for FiD models
            if "fid" in model.lower():
                base_model = "t5-large" if "large" in model.lower() else "t5-base"
                tokenizer = AutoTokenizer.from_pretrained(base_model)
            else:
                raise
        self.orig_kwargs = huggingface_pipeline_kwargs
        self.huggingface_pipeline_kwargs["tokenizer"] = tokenizer
        self.input_converter = FiDConverter(tokenizer.model_max_length)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.pipeline is None:
            model_name = self.huggingface_pipeline_kwargs.pop("model")
            model_instance = FiDforConditionalGeneration.from_pretrained(
                model_name, **self.orig_kwargs
            )
            self.huggingface_pipeline_kwargs["model"] = model_instance
            super(FiDGenerator, self).warm_up()

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        documents: List[Document],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self.pipeline is None:
            raise RuntimeError(
                "The generation model has not been loaded. Please call warm_up() before running."
            )
        if not prompt:
            return {"replies": []}
        
        # merge generation kwargs from init method with those from run method
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        
        # Input conversion using FiDConverter
        inputs = self.input_converter(self.pipeline.tokenizer, prompt, documents).to(
            self.pipeline.model.device
        )
        
        # First encode the 3D inputs using the FiD encoder
        with torch.no_grad():
            encoder_outputs = self.pipeline.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            )
        
        # Create 2D attention mask for generation by flattening the 3D mask
        batch_size, num_passages, seq_len = inputs["attention_mask"].shape
        encoder_attention_mask = inputs["attention_mask"].view(batch_size, num_passages * seq_len)
        
        # Generate from encoder outputs with scores
        gen_outputs = self.pipeline.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            stopping_criteria=self.stopping_criteria_list,
            return_dict_in_generate=True,  # Thêm tham số này
            output_scores=True,            # Thêm tham số này
            **updated_generation_kwargs,
        )
        
        # Lấy sequences và scores từ gen_outputs
        generated_tokens = gen_outputs.sequences[0]  # Sử dụng .sequences thay vì [0]
        replies = self.pipeline.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Bây giờ có thể truy cập scores
        scores = gen_outputs.scores if hasattr(gen_outputs, 'scores') else None
        
        if self.stop_words:
            # the output of the pipeline includes the stop word
            replies = [
                reply.replace(stop_word, "").rstrip()
                for reply in replies
                for stop_word in self.stop_words
            ]
        
        return {"replies": replies, "scores": scores}
