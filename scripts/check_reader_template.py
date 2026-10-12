"""Print the exact prompt a reader receives, after its chat template (tasks.md I4).

Run once per reader before the evaluation, on a machine with access to the model (the
Vistral repository is gated), and check that the system prompt, the [INST]/role markers
and the generation prompt look as the model card describes.

Usage:
  python scripts/check_reader_template.py --reader Viet-Mistral/Vistral-7B-Chat \
      --data data/tasks/vimmrc_dev.jsonl --task vimmrc \
      --system-prompt "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực."
"""
import argparse

from viword.data import load_task
from viword.eval import build_prompt, chat_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.reader)
    for example in load_task(args.task, args.data, args.n):
        messages = chat_messages(build_prompt(example, example.context), args.system_prompt)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        print("=" * 80)
        print(rendered)


if __name__ == "__main__":
    main()
