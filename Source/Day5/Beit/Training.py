from transformers import TrainingArguments, Trainer
from config import *
from Modeling import *
from PreData import *
args = TrainingArguments(
        f"finetuned-{name_weight}", # Tên thư mục để lưu kết quả huấn luyện
        remove_unused_columns=False,    # Giữ lại các cột dữ liệu trong tập dữ liệu ban đầu, kể cả cột không xài
        evaluation_strategy = "epoch",  # Đánh giá mô hình sau mỗi epoch
        save_strategy = "epoch",        # Lưu mỗi mô hình sau mỗi epoch
        learning_rate=lr,           # Cài đặt learning rate
        per_device_train_batch_size=batch_size, # Batch size train
        per_device_eval_batch_size=batch_size,  # Batch size test
        num_train_epochs=num_train_epochs,      # Cài đặt max epochs
        warmup_ratio=0.1,                 # Tỉ lệ warming up cho learning rate scheduler.
        logging_steps=5,                  # Số lượng bước cập nhật log thông tin trong quá trình huấn luyện.
        load_best_model_at_end=True,      # Tải mô hình tốt nhất (dựa trên metric đã chọn) sau khi hoàn thành quá trình huấn luyện.
        metric_for_best_model="accuracy", # Chọn metric để xác định mô hình tốt nhất.
        save_total_limit=2,               # Giới hạn số lượng checkpoints được lưu lại.
    )
trainer = Trainer(
    model,                       # Mô hình bạn muốn huấn luyện
    args,                        # Truyền các cấu hình mà bạn đã khai báo trước đó
    train_dataset=train_ds,      # Tập train dataset
    eval_dataset=test_ds,        # Tập evaluation dataset
    tokenizer=feature_extractor, # Feature extractor tương ứng với mô hình
    compute_metrics=compute_metrics, # Định nghĩa các tính metrics
    data_collator=collate_fn,        # Hàm tổ chức lại batch dữ liệu, trước khi được đưa vào mô hình
)

# Khởi chạy việc huấn luyện mô hình
train_results = trainer.train()
# Lưu lại mô hình vừa huấn luyện
trainer.save_model()

# Ghi lại các thông tin trong quá trình huấn luyện
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)

# Lưu lại trạng thái của trainer bao gồm: thông tin về epochs, các thông số khác, .. việc lưu giúp việc tiếp tục train trở nên đơn giản hơn
trainer.save_state()
## Đánh giá chất lượng của mô hình trên tập train và tập test

print("="*10+"evaluation in test dataset"+"="*10)
test_dict=trainer.evaluate(test_ds, metric_key_prefix='test')
print(test_dict)

print("="*10+"evaluation in test dataset"+"="*10)
train_dict=trainer.evaluate(train_ds, metric_key_prefix='train')
print(train_dict)
