import sacrebleu
from datasets import load_dataset
from src.translator import Translator
def main():
    dataset=load_dataset("wmt14", "de-en")["test"].select(range(50))
    src_texts=[ex["translation"]["en"] for ex in dataset]
    refs=[ex["translation"]["de"] for ex in dataset]
    translator=Translator("en", "de")
    preds=[translator.translate(t) for t in src_texts]
    bleu=sacrebleu.corpus_bleu(preds,[refs]).score
    print("BLEU score:",bleu)
if __name__=="__main__":
    main()
