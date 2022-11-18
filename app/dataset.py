import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import BatchEncoding, DefaultDataCollator, ViltProcessor


class ViltVQADataset(Dataset):
    def __init__(self, dataset: HFDataset, processor: ViltProcessor, label2id: dict[str, int]):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor
        self.label2id = label2id

        size = self.feature_extractor.size
        mean = self.feature_extractor.image_mean
        std = self.feature_extractor.image_std

        self.transform = T.Compose(
            [T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BatchEncoding:
        """
        item example:

        {'image_id': 2202312,
         'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480>,
         'category': '가전(가구)',
         'weather': '맑음',
         'questions': {'question_id': [7244561, 7244563, 7244559],
                       'question': ['흰색 테이블에 있는 검은색 의자는 2개야',
                        '싱크대 하부장에 달려 있는 손잡이는 유리 재질이야',
                        '수도꼭지 앞에 있는 개수대는 둥근 모양이야'],
                       'answer': ['아니요', '아니요', '아니요'],
                       'answer_type': ['긍정', '긍정', '긍정']}}
        """
        item = self.dataset[idx]
        pixel_values = self.transform(item["image"])
        filter_questions = self.filter_questions(item["questions"])
        num_questions = len(filter_questions["question"])

        if num_questions == 0:
            nidx = np.random.randint(0, len(self.dataset))
            return self[nidx]

        q_idx = np.random.randint(0, num_questions)
        question = filter_questions["question"][q_idx]
        question = self.add_question_mark(question)

        answer = filter_questions["answer"][q_idx]
        answer_id = self.label2id[answer]
        answer_id_tensor = torch.tensor(answer_id, dtype=torch.long)

        inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
        )
        inputs["pixel_values"] = pixel_values
        inputs["labels"] = answer_id_tensor
        return inputs

    @staticmethod
    def add_question_mark(text: str, p: float = 0.5) -> str:
        if not text.endswith("?") and np.random.rand() > p:
            text += "?"
        return text

    def filter_questions(self, questions: dict[str, list[str | int]]) -> dict[str, list[str | int]]:
        length = len(questions["question"])
        new = {k: [] for k in questions.keys()}
        for i in range(length):
            if questions["answer"][i] in self.label2id:
                for k, v in questions.items():
                    new[k].append(v[i])
        return new

class ViltVQADataModule(pl.LightningDataModule):
    def __init__(
        self, processor: ViltProcessor, batch_size: int = 32, num_workers: int = 12
    ):
        super().__init__()
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = DefaultDataCollator()

    def prepare_data(self) -> None:
        load_dataset("Bingsu/living_and_living_environment_based_vqa", split="train+validation", use_auth_token=True)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_dataset(
            "Bingsu/living_and_living_environment_based_vqa", split="train+validation", use_auth_token=True
        )
        label2id = self.load_label2id()
        train, valid = dataset.train_test_split(test_size=0.03, seed=42)
        self.train_dataset = ViltVQADataset(train, self.processor, label2id)
        self.valid_dataset = ViltVQADataset(valid, self.processor, label2id)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def load_label2id(self) -> dict[str, int]:
        root = Path(__file__).resolve().parent
        file = root / "label2id.json"
        with file.open("rb") as f:
            label2id = json.load(f)
        return label2id
