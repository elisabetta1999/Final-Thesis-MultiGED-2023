# Final-Thesis-MultiGED-2023

This repository contains all the notebooks used to align the development set of English (FCE and REALEC) and Italian of the MultiGED 2023 shared task to the original FCE, REALEC and MERLIN datasets in order to extract error types and corrections. This process is necessary to analyse in further detail the error types where my fine-tuned models fail more often in identifying incorrect tokens. The pre-processing notebooks can be found in **pre-processing-notebooks**.

The notebooks used to fine tune and generate predictions on the test set are stored in **fine-tuning-notebooks**. The notebooks to generate predictions on the aligned development sets are stored in evaluating-aligned-dev-notebooks. The fine tuned models can be found here:
[Google Drive folder.
](https://drive.google.com/drive/folders/1hujDAaKXmQoFQKJtkMPQ_wq1NAMABcrz?usp=drive_link.)

The notebooks used to extract quantitative information for my error analysis and to generate corrections through QWEN can be found in **error_analysis_stats_and_generating_corrections**.

Various .tsv files can also be found:

- processed_*.tsv are the result of the pre-processing steps and are used in the notebooks stored in evaluating-aligned-dev-notebooks.

- proc_*_xlm_roberta.tsv stored in error_analysis_stats_and_generating_corrections contain the aligned sentences together with XLM-RoBERTa's predictions and are used to extract statistics for the error analysis and then to generate corrections through Qwen.

- classified_sentences_*.tsv store all the manually labeled predictions and generations. They are not used in any notebook, but they collect the classifications used for the error analysis of both the discriminative and generative model.

To run the pre-processing notebooks, the fine-tuning and evaluating notebooks, one must also use the MultiGED-2023 datasets for English (FCE), English (REALEC) and Italian (MERLIN), together with the original datasets FCE, REALEC and MERLIN.

# Setup & Usage Guide
**Folder Structure**

For ease of use, unpack all files and notebooks from the GitHub repository into the same directory, without preserving subfolders. This means:

    All notebooks (pre-processing, fine-tuning, evaluation, error analysis) are in the same directory.

    All .tsv files (processed, predictions, classified sentences) are also in this directory.

    The thesis_utils.py file and requirements.txt are in the same directory.

    The folders of the original datasets (FCE, REALEC, MERLIN) and the MultiGED-2023 datasets should also be placed in this root directory.

**Important file placement:**

    proc_*_xlm_roberta.tsv files must be present in a directory called predictions_on_processedfiles to run the error analysis notebooks.
