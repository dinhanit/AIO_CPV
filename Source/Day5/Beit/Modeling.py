# Load pretrain Extractor
from transformers import AutoFeatureExtractor,AutoModelForImageClassification
from config import pretrain_name
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrain_name)
from PreData import *
import evaluate
import os
os.environ["WANDB_DISABLED"] = "true"
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


model_name = pretrain_name.split("/")[-1]
# Load model
model = AutoModelForImageClassification.from_pretrained(
    pretrain_name,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True,
)