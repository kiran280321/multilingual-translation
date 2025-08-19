import torch
from transformers import MarianMTModel, MarianTokenizer
from src.utils.language_codes import LANG_CODES
MODEL_NAME={
    ("en","de"):"Helsinki-NLP/opus-mt-en-de",
    ("de","en"):"Helsinki-NLP/opus-mt-de-en",
    ("en","fr"):"Helsinki-NLP/opus-mt-en-fr",
    ("fr","en"):"Helsinki-NLP/opus-mt-fr-en",
    ("en","es"):"Helsinki-NLP/opus-mt-en-es",
    ("es","en"):"Helsinki-NLP/opus-mt-es-en",
    ("en","hi"):"Helsinki-NLP/opus-mt-en-hi",
    ("hi","en"):"Helsinki-NLP/opus-mt-hi-en",
}
class Translator:
    def __init__(self,src_lang,tgt_lang):
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _load_model(self,src,tgt):
        if (src,tgt)not in MODEL_NAME:
            return None
        model_name=MODEL_NAME[(src,tgt)]
        tokenizer=MarianTokenizer.from_pretrained(model_name)
        model=MarianMTModel.from_pretrained(model_name).to(self.device)
        return tokenizer,model
    def _translate_once(self,text,src,tgt):
        pair=(src, tgt)
        if pair not in MODEL_NAME:
            raise ValueError(f"Model not available for {src} → {tgt}")
        tokenizer,model=self._load_model(src,tgt)
        inputs=tokenizer(text,return_tensors="pt",padding=True,truncation=True).to(self.device)
        translated=model.generate(**inputs,max_length=512)
        return tokenizer.decode(translated[0],skip_special_tokens=True)
    def translate(self, text: str)->str:
        if not text.strip():
            return "Please enter text to translate."
        if (self.src_lang,self.tgt_lang) in MODEL_NAME:
            return self._translate_once(text,self.src_lang,self.tgt_lang)
        elif (self.src_lang,"en")in MODEL_NAME and("en",self.tgt_lang) in MODEL_NAME:
            en_text=self._translate_once(text,self.src_lang,"en")
            return self._translate_once(en_text,"en",self.tgt_lang)
        else:
            raise ValueError(f"Translation path for {self.src_lang}->{self.tgt_lang} is not available.")
