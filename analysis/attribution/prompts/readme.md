
things to look at:
Write the user turn, sample from the model
1. handwrite prompts that would be refused, look at that
2. handwrite prompts that would be refused but with prompt jailbreaking (my grandma...)

- [ ] harmful queries ([HarmBench](https://huggingface.co/datasets/walledai/HarmBench)?)
- [ ] want queries where base and instruct model differ
- [ ] find adversarial suffixes where if you add it to the prompt, the model goes from refusing to accepting
- [ ] ideally, we can attribute the refusal behavior to this specific instruction tuning step and after instruction tuning it refuses a lot more

---
```sh
./sh/sbatch --gres=gpu:1 --constraint=48G --mem=128G --cpus-per-task=8 --partition=jag-standard --job-name=trace_iq ./run_on_gpu/run_trace_interesting_queries.sh
```
