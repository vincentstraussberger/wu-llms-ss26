# Project Report

**Applications of Data Science - Summer Term 2026**

Author: **Simon ERTL**

---
## Executive Summary

This project compares three different approaches for answering questions on Austrian Tax Law. The first model is a pure inference baseline, the second model is a parameter-efficient fine-tuned variant of the same base model, and the third model combines the fine-tuned model with retrieval-augmented generation (RAG). The goal was not only to generate legally plausible answers, but also to compare how much improvement can be achieved by supervised fine-tuning and by injecting external legal text at inference time.

All three models are built around the same base backbone, namely ``unsloth/gemma-3-4b-it-unsloth-bnb-4bit``. The models differ in the way task-specific knowledge is introduced. Model 1 uses the base model as-is. Model 2 adds LoRA-based supervised fine-tuning on an additional fine-tuning dataset. Model 3 keeps the fine-tuned model from Model 2 and supplements it with retrieved passages from Austrian legal texts.

The three models differ only in how task-specific knowledge is injected:

- Model 1 uses the backbone as-is and relies only on prompting.
- Model 2 applies LoRA-based supervised fine-tuning on a synthetic Austrian-tax-law dataset.
- Model 3 reuses the fine-tuned generator from Model 2 and adds retrieved legal passages from indexed PDF sources.

The main empirical finding is nuanced. If one prioritizes lexical overlap metrics that are closer to the project’s reference answer style, Model 3 performs best overall. It has the highest Token F1, BLEU-4, ROUGE-1, ROUGE-2, and ROUGE-L. If one prioritizes the semantic similarity metric BERTScore F1, Model 1 ranks first, most likely because its much longer answers cover more semantically related content, even when they are legally imprecise or overly verbose. Model 2 improves answer style and brevity, but underperforms Model 3 on nearly all overlap-based metrics.

## Data

In the following section, the data sources used in the project will be described. There are three main categories of data sources used in this project.

### Original Data

The central reference dataset is the Excel file ``Austrian Tax Law Dataset.xlsx``, sheet ``Dataset``. It contains 645 entries with at least the following relevant columns:
- ``id``: Unique identifier of the package. As the document was exported from Google Sheets, the following has to be noted: The evaluation uses a modified version of this document as the IDs for VAT-INTL topic were always the same; this was corrected and the corrected dataset was used for evaluation as the code merges references and generated answers based on the IDs.
- ``prompt``: A specific question about Austrian Tax Law corresponding to the chosen topic of the individual groups. These questions can be categorized into two categories:
    
    - **Real-Life cases**: These questions aim to depict a possible real-life case.
    - **General questions**: These questions do not involve possible scenarios and rather include questions asking about clarification about Austrian Tax Law

- ``correct_answer``: After the prompt was set, based on Austrian Legal Text the prompts were answered to form references for the evaluation during task 3.
- ``sources``: These column contains the sources that were referenced. These were mainly laws (e. g. EStG, UStG, …) and guidelines (e. g. EStR, …).

These entries form reference against which all three model outputs are evaluated. The prompts cover several tax-law subdomains, such as corporate tax, income tax, VAT, international taxation, real-estate tax law, and employment-related tax questions.

The file dataset_clean.csv is the input dataset that is actually fed into the model notebooks for answer generation. It contains 643 rows and only the columns ``id`` and ``prompt``. This means that the generation notebooks work on a slightly smaller set than the Excel reference dataset. The two missing IDs are ESTG27-015 and ESTG27-016, and these two entries consequently appear as missing answers in all three model outputs and in the evaluation.

### Fine-tuning data

As the second task involved fine-tuning an existing Large Language Model, additional data was needed as the model should not be fine-tuned on the original dataset as this would invalidate the results.
Therefore, using OpenAI’s ChatGPT, 300 additional packages were generated with the same structure as the original dataset. These 300 packages include prompts spanning across the topics of all groups.

### Retrieval Augmented Generation (RAG) data

As the third model involves retrieval augmented generation, additional data was needed again. This data would be injected into the prompt to provide the Large Language Model with additional, factual, up-to-date information.
For this, the following law's text was exported as PDFs from RIS: Bundesabgabenordnung, Einkommensteuergesetz, Körperschaftssteuergesetz, Unternehmensgesetzbuch, Umsatzsteuergesetz.

## Models

In the following section, the process in which the models were set up, fine-tuned, and used for prompt answering will be explained.

### Inference model

The inference model serves as the baseline for the full comparison. It uses the out-of-the-box Hugging Face model ``unsloth/gemma-3-4b-it-unsloth-bnb-4bit``. This is a quantized 4-bit version of Google’s Gemma 3 instruction model, wrapped for efficient loading with Unsloth. The notebooks use the model only for text input and text output, even though Gemma 3 is multimodal in principle. The model name itself already indicates the approximate model size of 4 billion parameters.

Gemma 3 was pretrained on a broad mix of web documents, code, mathematics, and images. For the present project, however, the pretrained model is not changed in Model 1. It is only prompted and used in deterministic inference mode.

A German system prompt is used to instruct the model to answer precisely, briefly, and in technically correct legal language, ideally in at most four to five sentences. The prompt also tells the model to admit uncertainty when necessary and not to invent legal sources.

The model uses greedy decoding. In other words, it always selects the most likely next token instead of sampling stochastically. This is useful for reproducibility and for a fair comparison across models, because repeated runs should lead to the same outputs.

The helper function
``build_prompt_question``
converts each raw question into the chat format expected by the tokenizer. The function ``answer_questions_batched``
then processes several questions at once. It applies left-padding, converts the chat messages into tokens via ``apply_chat_template``
, performs generation with ``model.generate(...)``
, and decodes only the newly generated part of the sequence. The key decoding choice is: ``do_sample = False``.

This means the model uses greedy decoding. It always selects the most likely next token instead of sampling (like top-p or top-k). This is useful for reproducibility and for a fair comparison across models, because repeated runs should lead to the same outputs.

From a methodological perspective, Model 1 forms the baseline. It bases it's answers only on the pretrained general knowledge of the backbone model and from the manually designed system prompt. It has no task-specific parameter update and no external retrieval step. This makes it a useful reference point for the later models.

### Fine-tuned model

The second model builds on the same Gemma-based backbone from Model 1, but now task-specific behavior is introduced by supervised fine-tuning with LoRA adapters. The notebook is split into two conceptual parts: first the actual fine-tuning, then a separate inference section that reloads the saved adapter and uses it to answer the benchmark questions. The notebook saves the LoRA adapter, this is necessary as the adapater is used for the third model.

The fine-tuning data is expected as sft_dataset.json. As explained above, the notebook requires the fields id, prompt, correct_answer, and sources. Several helper functions prepare this data:

- ``normalize_sources``: ensures that the sources field always becomes a clean list.
- ``build_assistant_text``: turns the generated reference into an assistant response and appends a bullet list of sources under a Quellen: heading if source information is available.
- ``render_training_text``: constructs a full chat conversation with system, user, and assistant messages and then converts it into a single training text using the tokenizer’s chat template.
- ``build_text_records``: assembles the final list of training records, and load_json(...) validates the JSON structure before conversion.

This preprocessing is highly relevant for understanding the fine-tuned model’s later output. The model is not trained on “free” answer generation, but on a fairly rigid answer template. As a consequence, Model 2 often answers in a short style and tends to reproduce the ``Quellen:`` block even when the legal references are weak or incorrect.

The core model and training hyper-parameters are as follows.

- ``base_model_name = unsloth/gemma-3-4b-it-unsloth-bnb-4bit``
- ``max_seq_len = 2048``
- ``load_in_4bit = True``
- ``rank r = 16``
- ``lora_alpha = 16``
- ``lora_dropout = 0.0``
- The following target modules are adapted: ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``gate_proj``, ``up_proj``, ``down_proj``. This means the notebook does not update all model weights. Instead, it inserts trainable low-rank matrices into key transformer sublayers. This is much more hardware-efficient than full fine-tuning.

- ``number of epochs: 1``
- ``learning rate: 2e-4``
- ``logging steps: 5``
- ``evaluation strategy: "epoch"``
- ``save strategy: "epoch"``
- ``validation split: 0.1``
- ``early stopping patience: 2``
- ``early stopping threshold: 0.0``

Altough early stopping is configured, this does in practice not come into effect as the number of epochs is set to 1. This was done as a higher number of epochs led to overfitting and unusable results.

The 10% validation split means that, for the 300-record fine-tuning dataset described in the report draft, approximately 270 examples are used for training and 30 for validation.

Another important output from the notebook is the number of trainable parameters. Only 32,788,480 of 4,332,867,952 total parameters are trained, which corresponds to roughly 0.76% of all parameters. This is typical for PEFT/LoRA and shows that the model was adapted efficiently.

After training, the adapter and tokenizer are saved to the folder ``model2_lora_adapter``. In the second half of the notebook, this adapter is loaded again and used for inference. The inference logic is very similar to Model 1, but one hyper-parameter differs: ``MAX_NEW_TOKENS = 256``. This limits the output length for this model. This was chosen as the model was trained to generate more concise answers.

Again, greedy decoding is used with ``do_sample = False``.

Model 2 attempts to specialize the general-purpose base model into a more Austrian-tax-law-oriented assistant. In practice, the results suggest that the model indeed learned a more compact and template-driven response style, but also that the fine-tuning setup was probably too weak to produce consistently strong legal answers. One epoch on 300 synthetic training items is enough to change the answer style, but apparently not enough to ensure robust legal grounding. The recurrent generic answers and repeated wrong paragraph patterns indicate that the model has partially learned the format of the training data, but not always the substance.

### RAG model

The third model combines retrieval and generation. Importantly, it does not use the plain base model for generation. Instead, it loads the saved LoRA adapter from Model 2. In that sense, Model 3 is not a completely separate architecture, but a fine-tuned generation model with an added retrieval pipeline.

The retrieval model is deliberately simple and transparent. The notebook uses:

- ``PdfReader`` from ``pypdf`` to read the legal PDFs
- ``TfidfVectorizer`` from ``scikit-learn`` for vectorization
- cosine similarity for ranking retrieved chunks

The preprocessing pipeline works as follows. The notebook reads all PDFs from a context directory. Each page is converted to text, and whitespace is normalized with the ``clean_text`` function. The text is then split into fixed-size word chunks using ``split_into_chunks`` with these parameters:

- ``CHUNK_SIZE_WORDS = 180``
- ``CHUNK_OVERLAP_WORDS = 40``

This produces overlapping text passages. The overlap is useful because legal arguments often span sentence boundaries. If a paragraph begins near the end of one chunk, part of it still remains visible in the next chunk.

The function ``build_rag_index`` stores each chunk together with its source file name and page number. It then computes TF-IDF vectors over all chunk texts with: ``ngram_range = (1, 2)``. This means the retriever considers both single words and two-word phrases. For legal text, this is sensible because many relevant expressions occur as short phrases rather than isolated tokens.

At retrieval time, ``retrieve_context(question, top_k = TOP_K_CHUNKS)`` transforms the question into a TF-IDF vector, computes cosine similarity to all indexed chunks, sorts the chunks by similarity, and formats the best hits into a context block that is appended to the user prompt.

The generation model itself is loaded from ``model2_lora_adapter``-folder, so the RAG model inherits the fine-tuned weights from Model 2. The generation settings are:

- ``MAX_SEQ_LENGTH = 2048``
- ``MAX_NEW_TOKEN = 512``
- ``BATCH_SIZE = 8``
- ``LOAD_IN_4_BIT = True``
- greedy decoding with ``do_sample = False``

The prompt construction differs from Model 1 and Model 2 because the question is combined with an explicit legal context block:

> Frage: ...
>
> Zusätzlicher Rechtskontext: ...

The system prompt also tells the model to prioritize the provided legal context. In theory, this should reduce hallucinations and improve source specificity.

The advantage of this model is that it can inject concrete legal passages into the prompt instead of relying only on the model’s internal memory. The disadvantage is also visible in the notebook. The retriever is lexical rather than semantic, and the context corpus is incomplete for several benchmark topics. This helps explain why Model 3 can be very strong on some KStG-, EStG-, and UStG-related questions, but still fail badly on topics that depend on laws outside the indexed corpus.

## Results

### Evaluation metrics

The evaluation notebook reads the reference dataset from ``Austrian Tax Law Dataset.xlsx`` and the three model outputs from their corresponding CSV files. For each model, the notebook merges the predictions with the reference answers by ``id`` using a left join. Missing answers are filled with empty strings. This is how the two missing predictions for ESTG27-015 and ESTG27-016 are handled.

The notebook defines several helper functions to normalize text, tokenize legal answers, compute overlap-based metrics, and classify typical errors.

The normalization step lowercases text, removes extra whitespace, and replaces non-breaking spaces. Tokenization uses a legal-text-oriented regular expression that captures section symbols, numbers, and German word tokens. This makes the later evaluation more suitable for legal answers than a completely naive whitespace split would be.

The exported metrics are the following:

- **Exact Match**:
This checks whether prediction and reference become exactly identical after normalization. It is a very strict metric and is naturally harsh in legal text generation, because even a correct paraphrase receives a score of zero.

 - **BLEU-4**:
This metric measures n-gram overlap between prediction and reference. The notebook applies smoothing because otherwise BLEU can easily collapse to zero on short answers.

- **ROUGE-1, ROUGE-2, ROUGE-L (F1)**:
These metrics measure lexical overlap on the unigram, bigram, and longest-common-subsequence level. The notebook uses the ``rouge_score`` package without stemming, which is a sensible choice for German legal language because aggressive stemming can distort meaning.

- **BERTScore Precision, Recall, and F1**:
These metrics compare prediction and reference semantically in embedding space. The notebook uses ``bert-base-multilingual-cased`` with ``lang = "de"`` and computes the scores on GPU. For this project, BERTScore is especially useful because legal answers can be correct even when they are phrased differently from the referenece.

- **Answer length in characters and tokens**
These descriptive statistics are not quality metrics in themselves, but they are highly informative in this project, because verbosity turns out to be one of the central qualitative differences between the models.

- **Token F1**:
Token F1 measures the overlap between the model answer and the corresponding reference answer on token level. In contrast to Exact Match, the generated answer does not have to be identical to the reference answer. Instead, the metric evaluates how many relevant tokens were produced by the model and how many relevant tokens from the reference answer were covered. Token F1 combines precision and recall into one value. Precision measures which share of the generated tokens also appears in the reference answer, while recall measures which share of the reference tokens is covered by the generated answer. The harmonic mean of both values is then calculated as the F1 score.

For the bonus error analysis, the notebook uses a rule-based ``classify_error`` function. The logic is transparent and easy to understand:

- **Exact match**
- **empty prediction** -> Missing answer
- **prompt leakage phrases such as “Ich verstehe. Du möchtest ...”** -> Prompt leakage / malformed output
- **repeated uncertainty phrases such as “Die Rechtslage ist unklar”** -> Overly generic uncertainty
- **answers longer than twice the reference length in tokens** -> Overly verbose answer
- **very low ROUGE-L and low BERTScore** -> Low semantic overlap
- **very low ROUGE-L and higher BERTscore** -> Possible paraphrase / low lexical overlap
- otherwise -> No common mistakes found

This is a useful basic error analysis, because it does not pretend to fully understand legal correctness, but it still captures the most obvious qualitative failure types.

### Evaluation Limitations

All three models have ``Exact Match = 0.0``. This does not mean that every answer is useless. It mainly shows that exact normalized string identity is too strict for this task. The models often paraphrase, change wording, or append source lines, which is enough to break exact match.

### Result Table

|Model                     |Rows in output|Rows evaluated|Exact Match|Token F1           |BLEU-4              |ROUGE-1 F1         |ROUGE-2 F1          |ROUGE-L F1         |BERTScore Precision|BERTScore Recall  |BERTScore F1      |Avg. answer length (chars)|Avg. answer length (tokens)|
|--------------------------|--------------|--------------|-----------|-------------------|--------------------|-------------------|--------------------|-------------------|-------------------|------------------|------------------|--------------------------|---------------------------|
|Model 1 - Inference       |643           |645           |0.0        |0.17885966791397917|0.018495787083715035|0.17395992370826435|0.047761631593224364|0.1114913179244329 |0.6366961473642394 |0.7140362331109453|0.672604454118152 |990.1984496124031         |135.09302325581396         |
|Model 3 - RAG (fine-tuned)|643           |645           |0.0        |0.18789164685446794|0.030880513181698353|0.1802519246646597 |0.05619247595114927 |0.13600558631236442|0.645931684139163  |0.6701311638188917|0.656827117167702 |443.43565891472866        |68.14108527131783          |
|Model 2 - Fine-tuning     |643           |645           |0.0        |0.17045418010190436|0.019098511392438275|0.1610371793465242 |0.04312853143131836 |0.1272591263161114 |0.6415928463603175 |0.6521480272906696|0.6455414323381675|359.5736434108527         |60.24186046511628          |

The same results can be found in the `evaluation_summary_ERTL.csv`-file.

Model 1 has the highest BERTScore Recall and the highest BERTScore F1. In other words, its answers are on average semantically closest to the gold references when broad semantic similarity is measured. However, the model is also by far the most verbose, with an average answer length of about 135 tokens. This suggests that part of its semantic score comes from long, broad answers that contain some relevant material, but also much unnecessary text.

Model 2 clearly shortens the answers compared to Model 1. This is an important stylistic improvement, and in fact it produces the shortest answers on average, at about 60 tokens. Nevertheless, its overlap-based metrics do not improve consistently, and its BERTScore F1 is the lowest of the three models. The fine-tuning therefore changed the model’s behavior, but not in a fully beneficial way.

Model 3 achieves the best Token F1, the best BLEU-4, the best ROUGE-1, the best ROUGE-2, and the best ROUGE-L. It also has the highest BERTScore Precision. This means that the retrieved context often helps the model produce more targeted and reference-like formulations. At the same time, its BERTScore F1 remains below Model 1.

### Error Analysis

For the error analysis, the generated answers were analyzed for possible coomon errors.

|model                     |error_category                   |count|
|--------------------------|---------------------------------|-----|
|Model 1 - Inference       |Overly verbose answer            |553  |
|Model 1 - Inference       |No common mistakes found         |87   |
|Model 1 - Inference       |Low semantic overlap             |2    |
|Model 1 - Inference       |Missing answer                   |2    |
|Model 1 - Inference       |Prompt leakage / malformed output|1    |
|Model 2 - Fine-tuning     |Overly generic uncertainty       |238  |
|Model 2 - Fine-tuning     |No common mistakes found         |203  |
|Model 2 - Fine-tuning     |Overly verbose answer            |146  |
|Model 2 - Fine-tuning     |Low semantic overlap             |56   |
|Model 2 - Fine-tuning     |Missing answer                   |2    |
|Model 3 - RAG (fine-tuned)|Overly generic uncertainty       |219  |
|Model 3 - RAG (fine-tuned)|No common mistakes found         |206  |
|Model 3 - RAG (fine-tuned)|Overly verbose answer            |143  |
|Model 3 - RAG (fine-tuned)|Low semantic overlap             |75   |
|Model 3 - RAG (fine-tuned)|Missing answer                   |2    |

The same results can be found in the `evaluation_error_ERTL.csv`-file.

Model 1 is dominated by the error category “Overly verbose answer.” This category occurs 553 times and is therefore by far the most important weakness of the baseline model. Model 1 usually does not fail by producing empty or malformed answers, but by generating responses that are much longer than the references and contain a large amount of unnecessary information. This observation fits the metric results well. Model 1 achieves the highest BERTScore F1, but it also has by far the longest answers on average. Its relatively strong semantic similarity therefore seems to be partly driven by broad and expansive responses that mention relevant concepts, even when they are not sufficiently precise.

Model 2 changes the error profile substantially. Its most frequent category is “Overly generic uncertainty,” which appears 238 times. This means that the fine-tuned model often responds in a vague or non-committal way instead of giving a concrete legal answer. At the same time, the number of “Overly verbose answer” cases drops to 146, so the fine-tuning clearly reduces the verbosity problem of the baseline. However, this stylistic improvement comes with a new cost. The model becomes more formulaic and often less informative. In addition, 56 cases are classified as “Low semantic overlap,” which indicates that some of the shorter answers are not only concise, but also incomplete or insufficiently aligned with the reference. Model 2 therefore improves the format of the answers more than their legal substance.

Model 3 shows a partly similar, but still distinct, error profile. Its most common category is again “Overly generic uncertainty,” with 219 cases, followed by 143 overly verbose answers. This means that the RAG model still inherits some of the cautious and template-like response behavior already visible in Model 2, although slightly less strongly. At the same time, Model 3 has 75 cases of “Low semantic overlap,” which is more than Model 2. This suggests that retrieval often helps the model produce more targeted and reference-like answers, but that retrieval failures can also introduce clearly incorrect or unhelpful context. In such cases, the model does not simply become vague; instead, it may become confidently misaligned with the reference. Model 3 is therefore often the most focused model, but also the one whose failures are most sensitive to the quality of the retrieved passages.

If one focuses on the dominant error type of each model, then the three systems do not fail in the same way. Model 1 mainly fails through excessive verbosity. Model 2 mainly fails through overly generic uncertainty. Model 3 also shows generic uncertainty as its largest category, but combines it with a noticeably higher number of low-semantic-overlap errors than Model 2. The error profiles are therefore clearly architecture-specific.

All three models contain exactly 2 missing answers, which is consistent with the two benchmark entries that are absent from all output files. Beyond that, however, the patterns differ markedly. Model 1 is characterized by long, diffuse answers, whereas Models 2 and 3 are more strongly shaped by the fine-tuned answer style and its tendency toward generic fallback formulations.

Overall, the error analysis supports the broader interpretation of the project. The baseline model tends to produce too much text, the fine-tuned model tends to become too cautious and formulaic, and the RAG model often produces the most targeted answers, but is also the most dependent on retrieval quality. For this reason, the models should not be seen as making the same kinds of mistakes. Rather, each architecture introduces its own characteristic failure mode: verbosity in Model 1, generic uncertainty in Model 2, and retrieval-sensitive misalignment in Model 3.

## Conclusion

This project demonstrates three progressively more specialized approaches to Austrian Tax Law question answering based on the same Gemma 3 4B backbone.

- Model 1 is a strong baseline in terms of general fluency, but it is far too verbose and often legally inaccurate.
- Model 2 shows that LoRA fine-tuning can strongly reshape answer style with very limited trainable capacity, but one epoch on 300 synthetic examples mainly teaches format, not robust legal reasoning.
- Model 3 is the most effective overall system for the actual benchmark task, because retrieval gives the model direct access to legal text and improves the overlap-based quality metrics substantially.

At the same time, the project also shows the limits of simple improvements:

- fine-tuning on templated synthetic data can create brittle citation behavior,
- lexical TF-IDF retrieval can help dramatically or mislead dramatically,
- aggregate semantic metrics like BERTScore can favor long but diffuse answers.

In a next iteration, the strongest improvements would likely come from:

- a cleaner and more diverse fine-tuning set
- a more optimized retrieval process (e. g. not extracting text from PDFs)
- a larger and more complete legal corpus
