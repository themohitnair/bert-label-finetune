import os
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

try:
    from datasets import Dataset
except Exception:
    Dataset = None

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

# Labels can be changed
LABELS = [
    "Inform",
    "Request",
    "Opine",
    "Promote",
    "Entertain",
    "Organize",
    "Motivate",
    "Greet",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def _ensure_datasets():
    if Dataset is None:
        raise RuntimeError("pip install datasets")


def _auto_columns(
    df: pd.DataFrame, text_col: Optional[str], label_col: Optional[str]
) -> Tuple[str, str]:
    if text_col is not None and label_col is not None:
        return text_col, label_col

    candidates_text = "description"
    candidates_label = "purpose"  # can change column name to be fine tuned on

    return candidates_text, candidates_label


def _map_labels(series: pd.Series) -> pd.Series:
    """
    Accepts either string labels from LABELS or int indices [0..7].
    """
    if series.dtype.kind in {"i", "u"}:
        bad = series[(series < 0) | (series >= len(LABELS))]
        if len(bad) > 0:
            raise ValueError(
                f"Found invalid numeric labels outside 0..{len(LABELS) - 1}"
            )
        return series.astype(int)
    unknown = sorted(set(series.unique()) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in data: {unknown}. Allowed: {LABELS}")
    return series.map(LABEL2ID).astype(int)


def _df_to_hf_dataset(df: pd.DataFrame) -> "Dataset":
    _ensure_datasets()
    df = df[["text", "label"]].reset_index(drop=True)
    return Dataset.from_pandas(df, preserve_index=False)


class PurposeClassifier:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = len(LABELS),
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def _tokenize(self, batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return self.tokenizer(
            batch["text"], truncation=True, padding=False, max_length=256
        )

    @staticmethod
    def _compute_metrics(eval_pred: Tuple):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        if _SKLEARN_OK:
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_score(labels, preds, average="macro"),
            }
        return {"accuracy": float((preds == labels).mean())}

    def train(
        self,
        train_ds: "Dataset",
        eval_ds: Optional["Dataset"] = None,
        output_dir: str = "./purpose-bert",
        lr: float = 2e-5,
        epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        gradient_accumulation_steps: int = 1,
        fp16: Optional[bool] = None,
        logging_steps: int = 50,
        save_total_limit: int = 2,
        seed: int = 42,
    ):
        _ensure_datasets()
        tokenized_train = train_ds.map(self._tokenize, batched=True)
        tokenized_eval = eval_ds.map(self._tokenize, batched=True) if eval_ds else None

        collator = DataCollatorWithPadding(self.tokenizer)
        if fp16 is None:
            fp16 = torch.cuda.is_available()

        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=weight_decay,
            eval_strategy="steps" if tokenized_eval is not None else "no",
            logging_steps=logging_steps,
            save_steps=logging_steps * 10,
            save_total_limit=save_total_limit,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            load_best_model_at_end=tokenized_eval is not None,
            metric_for_best_model="f1_macro" if _SKLEARN_OK else "accuracy",
            greater_is_better=True,
            report_to="none",
            seed=seed,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=self._compute_metrics
            if tokenized_eval is not None
            else None,
        )

        trainer.train()

        if tokenized_eval is not None and _SKLEARN_OK:
            preds = trainer.predict(tokenized_eval)
            y_true = preds.label_ids
            y_pred = preds.predictions.argmax(-1)
            print(
                classification_report(
                    y_true,
                    y_pred,
                    target_names=[ID2LABEL[i] for i in range(len(LABELS))],
                    digits=4,
                )
            )

        self.save(output_dir)
        return output_dir

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "labels.json"), "w") as f:
            json.dump({"labels": LABELS}, f)

    @classmethod
    def load(cls, model_dir: str) -> "PurposeClassifier":
        obj = cls.__new__(cls)
        obj.model_name = model_dir
        obj.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        obj.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        obj.device = "cuda" if torch.cuda.is_available() else "cpu"
        obj.model.to(obj.device)
        return obj

    @torch.no_grad()
    def predict(
        self, texts: List[str], batch_size: int = 64, return_probs: bool = False
    ) -> List[Dict[str, Any]]:
        self.model.eval()
        outputs: List[Dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).tolist()
            for j, text in enumerate(batch):
                rec = {
                    "description": text,
                    "purpose": ID2LABEL[pred_ids[j]],
                    "tone": "—",
                }
                if return_probs:
                    rec["probs"] = {
                        ID2LABEL[k]: float(probs[j, k]) for k in range(len(LABELS))
                    }
                outputs.append(rec)
        return outputs


def load_parquet_as_dataset(
    parquet_path: str,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> "Dataset":
    """
    Reads a single Parquet into a HuggingFace Dataset with columns: text, label (ints).
    Accepts label strings (will map) or ints 0..7.
    """
    _ensure_datasets()
    df = pd.read_parquet(parquet_path)
    t_col, y_col = _auto_columns(df, text_col, label_col)

    df = df.rename(columns={t_col: "text", y_col: "label"})
    df["label"] = _map_labels(df["label"]).astype(int)
    # Ensure text is string
    df["text"] = df["text"].astype(str)

    return _df_to_hf_dataset(df)


def train_from_parquet(
    train_parquet: str,
    eval_parquet: Optional[str] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    model_name: str = "bert-base-uncased",
    output_dir: str = "./purpose-bert",
    **train_kwargs,
) -> str:
    train_ds = load_parquet_as_dataset(
        train_parquet, text_col=text_col, label_col=label_col
    )
    eval_ds = (
        load_parquet_as_dataset(eval_parquet, text_col=text_col, label_col=label_col)
        if eval_parquet
        else None
    )

    clf = PurposeClassifier(model_name=model_name)
    clf.train(train_ds, eval_ds, output_dir=output_dir, **train_kwargs)
    return output_dir


def classify_and_store(
    texts: List[str],
    model_dir: str,
    parquet_filename: str = "output.parquet",
    store_module: str = "your_module_with_store_parquet",
    return_records: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Uses a saved model to classify `texts` and writes a parquet via your store_to_parquet().
    Set store_module to the python module path that contains store_to_parquet().
    """
    mod = __import__(store_module, fromlist=["store_to_parquet"])
    store_to_parquet = getattr(mod, "store_to_parquet")

    clf = PurposeClassifier.load(model_dir)
    records = clf.predict(texts, batch_size=64, return_probs=False)

    out = store_to_parquet(records, filename=parquet_filename)
    if out is None:
        return None
    return records if return_records else []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune a BERT classifier from a labeled Parquet file."
    )
    parser.add_argument(
        "--train_pq",
        type=str,
        required=True,
        help="Path to training Parquet (e.g., labelled.parquet)",
    )
    parser.add_argument(
        "--eval_pq", type=str, default=None, help="Optional eval Parquet"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default=None,
        help="Name of text column if not 'text'/'description'",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default=None,
        help="Name of label column if not 'label'/'purpose'",
    )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=32)
    args = parser.parse_args()

    train_from_parquet(
        train_parquet=args.train_pq,
        eval_parquet=args.eval_pq,
        text_col=args.text_col,
        label_col=args.label_col,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
    )
