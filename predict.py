from transformers import BertTokenizer, BartForConditionalGeneration


def prediction(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    outputs = model.generate(inputs['input_ids'], max_length=128)

    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return translated


if __name__ == '__main__':
    tokenizer_M2C = BertTokenizer.from_pretrained('final_model_M2C')
    tokenizer_C2M = BertTokenizer.from_pretrained('final_model_C2M')

    model_M2C = BartForConditionalGeneration.from_pretrained('final_model_M2C')
    model_C2M = BartForConditionalGeneration.from_pretrained('final_model_C2M')

    texts_M = ['你知道吗？', '景皇承受天运，继承大业。', '众人没有不服的。']
    tests_C = ['汝知之乎？', '景皇承天运，承大业。', '众莫不服。']

    translated_C = prediction(texts_M, tokenizer_M2C, model_M2C)
    translated_M = prediction(tests_C, tokenizer_C2M, model_C2M)

    print(translated_C)
    print(translated_M)
