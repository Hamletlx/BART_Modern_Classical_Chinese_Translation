from transformers import BertTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# 加载数据集
dataset = load_dataset('csv', data_files={'train': 'data_train.csv', 'test': 'data_test.csv'})

# 加载预训练模型和分词器
# 这里加载的是从HuggingFace下载的模型，然后对配置文件进行了一小部分修改，不是直接从HuggingFace下载
# 如果要从HuggingFace下载， 参数修改成：'fnlp/bart-base-chinese'，但可能运行代码会有警告
model = BartForConditionalGeneration.from_pretrained('bart_base_chinese')
tokenizer = BertTokenizer.from_pretrained('bart_base_chinese')


# 数据处理函数，此处input为现代文，target为文言文，即现代文翻译为文言文，可进行修改
def preprocess_function(examples):
    inputs = tokenizer(examples['modern'], max_length=128, truncation=True, padding='max_length')
    targets = tokenizer(examples['classical'], max_length=128, truncation=True, padding='max_length')
    inputs['labels'] = targets['input_ids']
    return inputs


# 应用数据处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 设置训练参数，可根据算力进行设置
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    eval_strategy='steps',
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# 使用 Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 保存模型和分词器
model.save_pretrained('final_model')
tokenizer.save_pretrained('final_model')
