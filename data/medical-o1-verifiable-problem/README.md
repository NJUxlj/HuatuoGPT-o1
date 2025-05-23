---
license: apache-2.0
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- medical
- biology
configs:
- config_name: default
  data_files:
  - split: train
    path: medical_o1_verifiable_problem.json
---
## Introduction

This dataset features open-ended medical problems designed to improve LLMs' medical reasoning. Each entry includes a open-ended question and a ground-truth answer based on challenging medical exams. The verifiable answers enable checking LLM outputs, refining their reasoning processes.

For details, see our [paper](https://arxiv.org/pdf/2412.18925) and [GitHub repository](https://github.com/FreedomIntelligence/HuatuoGPT-o1).


## Citation

If you find our data useful, please consider citing our work!
```
@misc{chen2024huatuogpto1medicalcomplexreasoning,
      title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs}, 
      author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
      year={2024},
      eprint={2412.18925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.18925}, 
}
```