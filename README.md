# 🚀 Job Titles Classification & Grouping

## 📌 Project Overview

This project tackles the problem of automatically classifying job titles by extracting job levels and job areas from raw job title text. Leveraging machine learning (supervised & unsupervised methods) and NLP techniques to build a pipeline that efficiently processes job titles, assigns hierarchical levels (e.g., Junior, Senior, Manager), and groups them into relevant industries (following NAICS classification).

### 🔹 Key Goals:
-	Process raw job title text and extract structured information.
-	Use pre-trained text embeddings to create a meaningful feature space.
-	Build classification models to predict job levels and job areas.
-	Package the solution in a Dockerized pipeline for easy deployment.

### 🔹 Dataset Used:

I used the dataset from Kaggle: Job Titles & Description, which consists of a single-column CSV with raw job titles.

---
## 📝 Project Description

In this project, keyword extraction was used to automatically generate labels for job titles, which were then used to train the classification model.

Predefined keyword mappings in **job-areas.json** and **job-levels.json** were used to assign job levels and job areas based on extracted terms from job titles.

This semi-supervised approach allowed for efficient and scalable dataset labeling, reducing the need for manual annotation.

Then I created meaningful embeddings using SentenceTransformers (Hugging Face) models.

After that I designed and trained a neural network classifier with Residual Blocks & Attention mechanisms for better contextual learning.

---
# 🏗 Model Architecture

The model is built using deep learning with a multi-output classification approach, predicting both job level and job area.

## 🔹 Steps in the Pipeline

### 1️⃣ Preprocessing
Cleaned job titles by removing symbols, HTML tags, and unnecessary characters.

Filtered out invalid job titles that contained only numbers, special characters, or irrelevant text.

### 2️⃣ Feature Extraction
Utilized multilingual SentenceTransformer embeddings (e.g., paraphrase-multilingual-mpnet-base-v2) to handle job titles in multiple languages.

Transformed job titles into a dense feature space, enabling effective classification of job levels and job areas.

### 3️⃣ Multi-Task Neural Network

A multi-task neural network was implemented to classify both job levels and job areas simultaneously. The model first processed job titles through shared layers, extracting deep semantic representations. To improve gradient flow and ensure stable training, residual blocks were incorporated. Additionally, attention mechanisms (SE Block) were used to emphasize important words within job titles, allowing the model to focus on the most relevant features.

The network had two output heads: one dedicated to predicting job levels (e.g., Student, Junior, Manager, C-Level) and another for classifying job areas (e.g., Engineering, Healthcare, Finance). This approach enabled the model to learn shared representations while optimizing for both classification tasks, improving overall performance.

**🔥 Why This Architechure is Suitable**

- Multi-task learning improves generalization by enabling the model to classify both job levels and job areas simultaneously.

- Residual connections enhance gradient flow, stabilizing training and improving convergence.

- SE Block (Squeeze-Excitation) dynamically reweights important words in job titles, increasing classification accuracy.

- SentenceTransformer embeddings effectively capture multilingual semantics, making the model robust to non-English job titles.

### 4️⃣ Training & Evaluation

The model was trained using weighted cross-entropy loss to address class imbalances in job level classification, ensuring better performance across both frequent and rare categories. 

To evaluate performance, F1-score was used as the primary metric, considering the multi-label nature of the classification task. Additionally, early stopping was implemented to monitor validation loss and prevent overfitting, ensuring the model generalizes well to unseen job titles.

### 5️⃣ Model Deployment: Docker Container & CLI for Production Use

To make the model production-ready, a Dockerized pipeline was implemented, allowing seamless execution in different environments.

**🛠 Docker Container Setup**

The training and inference scripts were packaged into a Docker container, ensuring consistent execution across different systems.

The model weights and encoders can either be preloaded inside the container or generated dynamically by running the training script within the container.

Multi-stage Docker builds were used to optimize image size and reduce unnecessary dependencies.

**⚡ Command-Line Interface (CLI) for Model Execution**

Build the Docker Image:

```bash
docker build -t job-predictor .
```

Train the model inside the container:

```bash
docker run --rm -v $(pwd)/data:/data job-predictor \
    --data_path /data/labeled_data.csv \
    --model_checkpoint /data/best_model.pth
```


Run inference on new job titles using the trained model:

```bash
docker run --rm -v $(pwd)/data:/data job-predictor \
    --data_path /data/job_titles.csv \
    --model_path /data/best_model.pth \
    --encoders_path /data/encoders.pkl \
    --output_path /data/predictions.csv
```

---
# 🏆 Project results

The model achieved high classification accuracy in predicting both job levels and job areas using a multi-label deep learning classifier. The results indicate that the model effectively captures job title semantics, although some challenges remain due to the automatic labeling process based on keyword extraction.

📊 Evaluation Metrics on Test Set:

| Metric |	Job Level Classification | Job Area Classification |
| -- | -- | -- |
| 🎯 Accuracy |	96.0%	| 78.0% |
| 🎯 Precision | 92.0%	| 65.0% |
| 🎯 Recall | 95.0% | 79.0% |
| 🎯 F1-Score |	96.0% |78.0% |


F1-score was used as the primary metric since job titles can belong to multiple categories (multi-label classification).

The model performs exceptionally well for job level classification (96% F1-score), suggesting that hierarchical patterns in job titles are well understood.

Job area classification shows slightly lower performance (78% F1-score), indicating potential room for improvement in domain-specific categorization.

**🔍 Impact of Keyword-Based Labeling**

The labeling process relied on keyword extraction from job titles, mapping them to predefined job levels and job areas. While this method enabled efficient data annotation, it introduced some potential limitations:
- Automated labeling scales well, reducing the need for manual annotation.
- Some job titles may have been incorrectly mapped due to ambiguous keywords.
- Overlapping job areas (e.g., “Data Engineer” could be classified under both IT and Engineering) could introduce noise in training.


# 🚀 Potential Improvements

Despite the strong results, further refinements could enhance the model’s performance:

✅ Refining keyword mappings to reduce ambiguity in label assignment.

✅ Enhancing embeddings by fine-tuning SentenceTransformer models on a larger job title dataset.

✅ Hierarchical classification (first predicting broad categories, then refining into subcategories) could improve job area accuracy.

✅ Incorporating external job taxonomies (e.g., O*NET, NAICS) for a more structured classification system.

✅ Deploying as a FastAPI microservice for real-time classification.

✅ Integrating model monitoring (e.g., Prometheus, Grafana) to track inference accuracy in production.


# ⚙️ Local Setup Instructions

🛠 Setting Up Poetry (Local Development)

1️⃣ Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2️⃣ Clone the repository:

```bash
git clone https://github.com/lezhocheck/job-titles-grouping.git
cd job-titles-grouping
```

3️⃣ Install dependencies:

```bash
poetry install
```

4️⃣ Run training:

```bash
poetry run python main_train.py --data_path ./data/labeled_data.csv --model_checkpoint ./models/best_model.pth
```

5️⃣ Run inference:

```bash
poetry run python main_inference.py --data_path ./data/job_titles.csv --model_path ./models/best_model.pth --encoders_path ./models/encoders.pkl --output_path ./data/predictions.csv
```