---
language:
- en
license: mit
task_categories:
- conversational
- text-classification
- token-classification
- table-question-answering
- question-answering
- zero-shot-classification
- summarization
- feature-extraction
- text-generation
- text2text-generation
pretty_name: SlimOrca
size_categories:
- 100K<n<1M
---

# Overview

This is a new curated subset of our OpenOrca data. This release provides an efficient means of reaching performance on-par with using larger slices of our data, while only including ~500k GPT-4 completions.

The key change in this dataset is that we've done an additional pass, using GPT-4 to remove answers which appear wrong based on the human annotations from the FLAN dataset.
This reduces the dataset size to only ~500k entries, allowing training to a similar quality level to our previous releases with 2/3 the compute requirement.


# Demo Models

* https://huggingface.co/openaccess-ai-collective/jackalope-7b
* https://huggingface.co/Open-Orca/Mistral-7B-SlimOrca


# Citation

```bibtex
@misc{SlimOrca,
  title = {SlimOrca: An Open Dataset of GPT-4 Augmented FLAN Reasoning Traces, with Verification},
  author = {Wing Lian and Guan Wang and Bleys Goodson and Eugene Pentland and Austin Cook and Chanvichet Vong and "Teknium"},
  year = {2023},
  publisher = {HuggingFace},
  url = {https://https://huggingface.co/Open-Orca/SlimOrca}
}
```
```bibtex
@misc{mukherjee2023orca,
      title={Orca: Progressive Learning from Complex Explanation Traces of GPT-4}, 
      author={Subhabrata Mukherjee and Arindam Mitra and Ganesh Jawahar and Sahaj Agarwal and Hamid Palangi and Ahmed Awadallah},
      year={2023},
      eprint={2306.02707},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```bibtex
@misc{longpre2023flan,
      title={The Flan Collection: Designing Data and Methods for Effective Instruction Tuning}, 
      author={Shayne Longpre and Le Hou and Tu Vu and Albert Webson and Hyung Won Chung and Yi Tay and Denny Zhou and Quoc V. Le and Barret Zoph and Jason Wei and Adam Roberts},
      year={2023},
      eprint={2301.13688},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```