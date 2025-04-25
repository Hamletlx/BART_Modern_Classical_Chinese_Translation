# pip install Flask

from flask import Flask, render_template_string, request
from transformers import BertTokenizer, BartForConditionalGeneration
import torch

# 使用 render_template_string 的方法来进行渲染
html = """
<h1>现代文 ↔ 文言文 转换器</h1>
<form method="post">
    <textarea name="text" rows="6" cols="60" placeholder="请输入文本">{{ text }}</textarea><br><br>
    <select name="direction">
        <option value="m2c" {% if direction == "m2c" %}selected{% endif %}>现代文 → 文言文</option>
        <option value="c2m" {% if direction == "c2m" %}selected{% endif %}>文言文 → 现代文</option>
    </select><br><br>
    <button type="submit">翻译</button>
</form>
{% if result %}
<h2>翻译结果：</h2>
<div>{{ result }}</div>
{% endif %}
"""

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_m2c = BartForConditionalGeneration.from_pretrained("model_m2c").to(device)
tokenizer_m2c = BertTokenizer.from_pretrained("model_m2c")

model_c2m = BartForConditionalGeneration.from_pretrained("model_c2m").to(device)
tokenizer_c2m = BertTokenizer.from_pretrained("model_c2m")

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
    translated = [s.replace(' ', '') for s in translated]
    return translated

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    text = ""
    direction = "m2c"

    if request.method == "POST":
        text = request.form.get("text", "")
        direction = request.form.get("direction", "m2c")

        if direction == "m2c":
            result = prediction([text], tokenizer_m2c, model_m2c)[0]
        else:
            result = prediction([text], tokenizer_c2m, model_c2m)[0]

    return render_template_string(html, result=result, text=text, direction=direction)

if __name__ == "__main__":
    app.run(debug=True)
