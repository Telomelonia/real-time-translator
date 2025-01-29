# translate.py
import multiprocessing
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Translator:
    def __init__(self):
        print("Loading translation model...")
        model_name = "Helsinki-NLP/opus-mt-ja-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        
        # Optimize model for inference
        self.model.eval()  # Set to evaluation mode
        if torch.cuda.is_available():
            self.model = self.model.to("cuda").half()  # Use half precision
        
        # Enable torch optimizations
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Translation model loaded!")

    @torch.inference_mode()  # Faster than no_grad
    def translate(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=2,  # Reduced from default 4
                length_penalty=0.6,
                early_stopping=True
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

def translation_process(queue):
    try:
        translator = Translator()
        print("Waiting for text to translate...")
        
        # Buffer for batch processing
        buffer = []
        
        while True:
            try:
                item = queue.get()
                if item is None:
                    break
                    
                translation = translator.translate(item['text'])
                print(f"\n{item['timestamp']}")
                print(f"🇯🇵: {item['text']}")
                print(f"🇬🇧: {translation}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Translation error: {e}")
                continue
    except KeyboardInterrupt:
        print("\nTranslation process stopped")

def main():
    try:
        # Use higher priority for translation process
        if os.name == 'nt':  # Windows
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        
        translation_queue = multiprocessing.Queue()
        
        translator_process = multiprocessing.Process(
            target=translation_process, 
            args=(translation_queue,)
        )
        translator_process.start()
        
        import transcribe
        transcribe.main(translation_queue)
        
        translator_process.join()
    except KeyboardInterrupt:
        print("\nStopping all processes...")

if __name__ == "__main__":
    main()