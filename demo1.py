# pip install PySide6

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QTextEdit, QPushButton, QMessageBox, QComboBox
)
import sys
import torch
from transformers import BertTokenizer, BartForConditionalGeneration


class TranslatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("现代文-文言文 转换器")
        self.setMinimumSize(300, 200)
        self.resize(400, 300)

        layout = QVBoxLayout()

        self.combo = QComboBox()
        self.combo.addItems(["现代文 -> 文言文", "文言文 -> 现代文"])
        layout.addWidget(self.combo)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("请输入文本")
        layout.addWidget(self.input_edit)

        self.button = QPushButton("翻译")
        layout.addWidget(self.button)

        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlainText("翻译结果将在此显示")
        layout.addWidget(self.output_edit)

        self.setLayout(layout)

        self.button.clicked.connect(self.translate)

        # 配置默认使用GPU，如果没有就使用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_m2c = BartForConditionalGeneration.from_pretrained('model_m2c').to(self.device)
        self.tokenizer_m2c = BertTokenizer.from_pretrained('model_m2c')

        self.model_c2m = BartForConditionalGeneration.from_pretrained('model_c2m').to(self.device)
        self.tokenizer_c2m = BertTokenizer.from_pretrained('model_c2m')

    def translate(self):
        text = self.input_edit.toPlainText()
        if text:
            direction = self.combo.currentText()
            if direction == "现代文 -> 文言文":
                result = self.prediction([text], self.tokenizer_m2c, self.model_m2c)[0]
            else:
                result = self.prediction([text], self.tokenizer_c2m, self.model_c2m)[0]
                pass
            self.output_edit.setPlainText(result)
        else:
            QMessageBox.warning(self, "输入错误", "请输入要翻译的文本")

    def prediction(self, texts, tokenizer, model):
        inputs = tokenizer(
            texts,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)
        outputs = model.generate(inputs['input_ids'], max_length=128)
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated = [s.replace(' ', '') for s in translated]
        return translated

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TranslatorApp()
    window.show()
    sys.exit(app.exec())
