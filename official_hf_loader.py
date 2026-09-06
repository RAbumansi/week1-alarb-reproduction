"""Official ALARB loading example.

Source: https://huggingface.co/datasets/THIQAH-RD/ALARB
Dataset license: Apache-2.0.
"""

from datasets import load_dataset


dataset = load_dataset("THIQAH-RD/ALARB")
print(dataset)

