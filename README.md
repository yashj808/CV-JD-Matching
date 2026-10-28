# Resume-Job_Description(CV-JD)-Matching

- **Objective:**
    - Build a PDF extractor to pull relevant details from CVs in PDF format, and match them against the job descriptions from the Hugging Face dataset.

- **Dataset Used**
    - [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
    - [Job Descriptions from Hugging Face](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions/viewer/default/train?row=0)

- **PDF Extractor**(`01_pdf-data-extraction.ipynb`)
    - PDF Extraction was done using `pdfplumber` library
    - Python `regex` was used to extract `Skills` and `Education` Sections from the extracted text.
    - Finally the relevent information was stores in a csv file `pdf_extracted_skills_education.csv`
    
    - **Challenges Faced:** Using `regex` I could extract these `Skills` and `Education` part. But it was hard to generalise this over resumes of different format. So, more research will be needed to efficently extract these. Extracting `Experience` was really tough, I couldn't think of any `regex` that could extract `Company_Name, Start_Date, End_Date` within multiple headers(i.e., people with multiple experiences)
    
    - **Propesed Solution:** One thing that I know is training **Custom Named Entity Recogniton(NER)** but for this we need a custom tagged dataset for skills, Education, and Experience.


- **CV-JD Matching**(`03_cv-jd-matching.ipynb`)
    - Got the JD dataset from hugging face `datasets` library. 15 JDs from the dataset were selected for this project.
    - Basic text cleaning like lower_case, removing punctuations/emails/phone_numbers was done on the extracted resumes.
    - Tokenization and Embeddings for JDs & CVs were created using `DistilBertTokenizer, DistilBertModel` from `transformers` library.
    - For matching CV-JD, `cosine_similarity` was used from `sklearn` library.
    - Finally, Top-5 Candidates were extracted for the respective Job Descriptions, according to the respective similarity score.
    
- **Evaluation & Confusion Matrix Analysis**(`04_evaluation-accuracy-confusion-matrix.ipynb`)
    - Evaluates accuracy across three primary dimensions:
      1. **Multiclass Resume Categorization**: Supervised benchmark evaluating classification accuracy and a complete $24 \times 24$ Confusion Matrix (raw counts and normalized recall heatmaps).
      2. **Zero-Shot Semantic Matching**: Matching resumes against domain prototypes via cosine similarity with confusion matrix visualization.
      3. **Information Retrieval (IR) Accuracy**: Hit@1, Hit@3, Hit@5 (73.3%), and Mean Reciprocal Rank (MRR) for candidate retrieval.
      4. **Binary Match Confusion Matrix**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN), ROC curve (AUC > 0.77), and threshold tuning curve.
    

 - **Overall Challenges Faced:**
     - This was my first time working with `PyTorch` and `transformers` library
     - More than Modelling, Extracting the neccessary data is more tough as mentioned above in **PDF Extractor**.
