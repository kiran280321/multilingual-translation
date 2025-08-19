# 🌍 Multilingual Translation App

A web-based multilingual translation system built with Streamlit and Hugging Face models.
The app allows users to translate customer queries across multiple languages, fine-tune models on WMT datasets, and evaluate translation quality using BLEU score.

---

## 🚀 Features

* 🌐 Translate queries between multiple languages
* ⚡ Fine-tune models on WMT dataset
* 📊 Evaluate translation quality with BLEU
* 🎨 Interactive Streamlit interface
* ☁️ Deployable on Streamlit Cloud

---

## 📁 Project Structure

```
multilingual-translation/
├── app/
│   └── streamlit_app.py       # Streamlit web app
├── api/
│   └── main.py                # FastAPI backend
├── src/
│   ├── translator.py          # Translation logic
│   └── utils/
│       └── language_codes.py  # Language code mappings
├── training/
│   └── finetune_nllb_lora.py  # Fine-tuning script
├── evaluation/
│   └── evaluate_bleu.py       # BLEU score evaluation
├── requirements.txt           # Dependencies
├── Dockerfile                 # For containerization
└── README.md                  # Documentation
```

---

## 🛠 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/kiran280321/multilingual-translation.git
cd multilingual-translation
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Fine-tune the model**

```bash
python training/finetune_nllb_lora.py
```

4. **Run the app**

```bash
streamlit run app/streamlit_app.py
```

The app will be available at:
👉 [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deployment

* **GitHub Repo:** [Multilingual Translation](https://github.com/kiran280321/multilingual-translation)
* **Live Demo:** [Streamlit Cloud App](https://multilingual-translation.streamlit.app)

---

## 📊 Evaluation

The fine-tuned model was evaluated using BLEU score on the test dataset.

**Result:**

```
BLEU score: 28.5
```

This indicates that the model achieves competitive translation quality on **English → German**.

---

## 🧠 Skills Tested

* **Natural Language Processing (NLP)**
* **Sequence-to-Sequence Models**
* **Model Deployment (Streamlit & FastAPI)**

---

## 📃 License

This project is licensed under the MIT License.
You are free to use, modify, and share it.

---

## 🙋‍♂️ Author

Made with ❤️ by **KIRAN280321**
