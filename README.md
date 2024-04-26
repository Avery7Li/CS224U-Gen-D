## Gen-D: Analysis Framework for Gender-specific Information Distribution

### :thought_balloon: Research Question

Social biases in language models have widely raised public attention. Current mitigations in- volve correcting biases in word embeddings, generating gender-neutral datasets, and debiasing models as a whole. Each mitigation comes with drawbacks – word embeddings fall short of ensuring sentence-level fairness, and debiasing models as a whole using gender- neutral datasets induces burden on data col- lection. 

In this project, we propose an analysis framework to characterize gender-specific information distribution for general language models. Our findings include that **fine-tuning attention layers only**, instead of the whole model, using anti-bias datasets is enough for debiasing, and gender-specific information is concentrated in a small portion of model parameters. Our proposed method has great potential in reducing cost for collecting large gender- neutral data and providing insights for effective model debiasing by targeting specific layers.

### Data 

Our analysis focused on the [BUG](https://github.com/slab-nlp/bug) dataset (Levy et al. 2021), a gender bias dataset assciated with tasks of coreference resolution. The full dataset contains 105,687 sentences in total with a human entity which is identified by their profession and a gendered pronoun. The sen- tences are classified as anti-stereotype, neutral, and stereotype sentences based on their correspond- ing profession and pronoun. For simplicity, we filtered the dataset and focused on the 64,299 sentences that only contain *one pronoun*. Among the filtered sentences, 1,113 sentences are validated by human and have a "gold quality". Please refer to [datasets.py](https://github.com/Avery7Li/CS224U-Gen-D/blob/main/datasets.py) for processing code. 

The dataset was further split on the stereotype attribute into a *stereotypical dataset* and a *anti-stereotypical dataset*.

### Models Evaluted

1. **BERT** (bert-base-uncased)
2. **DistillBERT** (distilbert-base-uncased)



### Bias Evaluation

The core task for our models can be framed as masked word prediction. For each sentence, we replace its pronoun with the "[MASK]" token, and treat the pronoun as the gold label. For example:

• *input*: "Among them was the president himself"

• *output*: "Among them was the president [MASK]"

• *label*: "himself"

A language model will predict the probability of the masked word given the rest of the sentence. A biased model may assign higher probability to *himself* than to *herself*. We measure the model’s gender bias by comparing the conditional probability of the stereotypical output versus the conditional probability of the anti stereotypical output.

### Investigating Gender Bias Distribution
BERT like models have several attention blocks, and we sought to explore which block(s) includes the most gender-related information through **finetuning attention blocks** both *individually* and *cumulatively*, and compare its performance to the original model. To achieve this, we first finetuned language models on anti-stereotypical texts. Please refer to [finetue.py](https://github.com/Avery7Li/CS224U-Gen-D/blob/main/finetune.py) for finetuning code.

#### Individual Intervention

For the k-th transformer layers (k = 1,..., 6), the attention weights of the vanilla models were replaced by the attention weights from the finetuned models while parameters from other attention blocks were not changed. 

#### Cumulative Intervention

For the kth cumulative intervention model, the weights in the first k attention blocks were all replaced by those from the fine-tuned models. Please refer to [intervene.py](https://github.com/Avery7Li/CS224U-Gen-D/blob/main/intervene.py) for intervening code.

