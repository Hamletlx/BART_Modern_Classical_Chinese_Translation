import torch
from transformers import BertTokenizer, BartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    outputs = model.generate(inputs['input_ids'], max_length=128)
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translated = [s.replace(' ', '') for s in translated]  # 将每个字符串中的空格去掉
    return translated


if __name__ == '__main__':
    # 需要将训练好的模型权重参数放置在model_m2c和model_c2m文件夹中

    # 测试数据
    texts_m = ['你知道吗？', '景皇承受天运，继承大业。', '众人没有不服的。']
    tests_c = ['汝知之乎？', '景皇承天运，承大业。', '众莫不服。']

    # 现代文转换成文言文
    model_m2c = BartForConditionalGeneration.from_pretrained('model_m2c').to(device)
    tokenizer_m2c = BertTokenizer.from_pretrained('model_m2c')
    translated_c = prediction(texts_m, tokenizer_m2c, model_m2c)
    print(translated_c)

    # 文言文转换成现代文
    model_c2m = BartForConditionalGeneration.from_pretrained('model_c2m').to(device)
    tokenizer_c2m = BertTokenizer.from_pretrained('model_c2m')
    translated_m = prediction(tests_c, tokenizer_c2m, model_c2m)
    print(translated_m)
