# translate.py
import multiprocessing
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Translator:
    def __init__(self):
        print("Loading translation model...")
        model_name = "Helsinki-NLP/opus-mt-ja-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        print("Translation model loaded!")

    def translate(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            translated_ids = self.model.generate(**inputs, max_length=128)
            translation = self.tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
            return translation
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails

def translation_process(queue):
    try:
        translator = Translator()
        print("Waiting for text to translate...")
        
        while True:
            try:
                item = queue.get()
                if item is None:  # Stop signal
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
        # Create a queue for receiving text
        translation_queue = multiprocessing.Queue()
        
        # Start translation process
        translator_process = multiprocessing.Process(
            target=translation_process, 
            args=(translation_queue,)
        )
        translator_process.start()
        
        # Start transcription process
        import transcribe
        transcribe.main(translation_queue)
        
        # Clean up
        translator_process.join()
    except KeyboardInterrupt:
        print("\nStopping all processes...")

if __name__ == "__main__":
    main()