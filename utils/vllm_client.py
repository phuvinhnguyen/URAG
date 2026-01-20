import os
import multiprocessing as mp
import requests
import time
import subprocess
from typing import List
from transformers import AutoTokenizer

os.environ['VLLM_ENGINE_MP_START_METHOD'] = 'spawn'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

DEFAULT_VLLM_PORT = 8000
DEFAULT_HOST = "127.0.0.1"
VLLM_SERVER_PROCESS = None

class LLMClient:
    def __init__(self, model_name: str = 'Orion-zhen/Qwen3-8B-AWQ', use_4bit: bool = True, port: int = None):
        self.model_name = model_name
        self.port = port or int(os.getenv('VLLM_PORT', DEFAULT_VLLM_PORT))
        self.host = os.getenv('VLLM_HOST', DEFAULT_HOST)
        self.base_url = f"http://{self.host}:{self.port}"
        self.server_started = False
        
        # Load tokenizer for chat template support
        print(f"Loading tokenizer for {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.has_chat_template = (
                hasattr(self.tokenizer, 'chat_template') and 
                self.tokenizer.chat_template is not None
            )
            if self.has_chat_template:
                print(f"✅ Chat template detected. Will auto-format user prompts.")
            else:
                print(f"⚠️ No chat template found. Using raw prompts.")
        except Exception as e:
            print(f"⚠️ Failed to load tokenizer: {e}. Using raw prompts.")
            self.tokenizer = None
            self.has_chat_template = False
        
        # Check if server is running
        if not self._is_server_running():
            print(f"🚀 No vLLM server found. Starting on {self.host}:{self.port}...")
            self._start_server()
            self.server_started = True
        else:
            print(f"✅ Connected to existing vLLM server at {self.host}:{self.port}")
        
        # Verify model is loaded
        if not self._verify_model():
            raise RuntimeError(f"Model {model_name} not loaded on server")
    
    def _is_server_running(self) -> bool:
        """Check if vLLM server is responding"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _verify_model(self) -> bool:
        """Verify the correct model is loaded"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                for model in models:
                    if model.get("id") == self.model_name:
                        return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Cannot verify model: {e}")
            return False
    
    def _start_server(self):
        """Launch vLLM server as background process"""
        # Build command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", "1",
            "--dtype", "float16",
            "--trust-remote-code"
        ]
        
        if "AWQ" in self.model_name:
            cmd.extend(["--quantization", "AWQ"])
        
        # Start server in background
        print(f"Running: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for server to be ready
        max_retries = 120
        for i in range(max_retries):
            if self._is_server_running() and self._verify_model():
                print("✅ Server ready!")
                return
            time.sleep(1)
            if i % 10 == 0:
                print(f"⏳ Waiting for server to start... ({i}s)")
        
        proc.terminate()
        raise RuntimeError("Failed to start vLLM server within timeout")
    
    def _apply_chat_template(self, prompt: str) -> str:
        if not self.has_chat_template:
            return prompt
        
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # TODO: modify this for reasoning or pure generation task
            )
        except Exception as e:
            print(f"⚠️ Failed to apply chat template: {e}. Using raw prompt.")
            return prompt

    def call_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 30000,
        temperature: float = 0.1,
        **sampling_kwargs
    ) -> List[str]:
        if not prompts: return []
        
        formatted_prompts = [self._apply_chat_template(p) for p in prompts]
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": formatted_prompts,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                **sampling_kwargs
            }
            
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload
            )
            response.raise_for_status()
            
            results = []
            for choice in response.json()["choices"]:
                text = choice["text"].strip()
                if '</think>' in text:
                    text = text.split('</think>')[-1].strip()
                results.append(text)
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API call failed: {e}")
            return [""] * len(prompts)
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            return [""] * len(prompts)

    def call(self, prompt: str, **kwargs) -> str:
        return self.call_batch([prompt], **kwargs)[0]
    
    def shutdown(self):
        try: requests.post(f"{self.base_url}/shutdown", timeout=2)
        except: pass

if __name__ == "__main__":
    client = LLMClient(model_name="Orion-zhen/Qwen3-8B-AWQ")
    print(client.call("Hey, Introduce yourself"))