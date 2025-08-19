from fastapi import FastAPI
from pydantic import BaseModel
from src.translator import Translator
app=FastAPI(title="Multilingual Translator API")
class TranslateReq(BaseModel):
    text:str
    src_lang:str
    tgt_lang:str
@app.post("/translate")
def translate(req:TranslateReq):
    try:
        translator=Translator(req.src_lang,req.tgt_lang)
        output=translator.translate(req.text)
        return{"translation":output}
    except Exception as e:
        return{"error":str(e)}
