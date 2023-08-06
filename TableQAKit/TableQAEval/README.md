# TableQAEval

<p align="center">
<img src="../../figs/TableQAEval.png" width="400">
</p>

TableQAEval is a benchmark to evaluate the performance of LLM for TableQA. It evaluates LLM's modeling ability of long tables (context) and comprehension capabilities (numerical reasoning, multi-hop reasoning).

## Leaderboard

| Model | Parameters | Numerical Reasoning | Multi-hop Reasoning | Structured Reasoning | Total |
| ---   | ---        | ---                 | ---                 | ---                  | ---   |
| Turbo-16k-0613 | -     | 20.3     | 52.8 | 54.3 | 43.5 |
| LLaMA2-7b-chat | 7B | 2.0 | 14.2 | 13.4 | 12.6 |
| ChatGLM2-6b-8k | 6B | 1.4 | 10.1 | 11.5 | 10.2  |
| LLaMA2-7b-4k | 7B | 0.8 | 9.2 | 5.4 | 6.6 |
| longchat-7b-16k | 7B | 0.3 | 7.1 | 5.1 | 5.2 |
| LLaMA-7b-2k | 7B | 0.5 | 7.3 | 4.1 | 4.5 |
| MPT-7b-65k | 7B | 0.3 | 3.2 | 2.0 | 2.3 |
| LongLLaMA-7b-2k | 7B | 0.0 | 4.3 | 1.7 | 2.0 |