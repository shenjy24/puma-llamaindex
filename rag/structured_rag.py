import json
import os
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, Field
from llama_index.readers.file import PDFReader
from pathlib import Path
from llama_index.llms.deepseek import DeepSeek

# 加载 .env 文件中的环境变量
# 这行代码会查找当前目录下的 .env 文件并将变量注入到 os.environ 中
load_dotenv()

# 从环境变量获取配置, 建议设置默认值作为兜底，防止 .env 漏配
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

llm = DeepSeek(
    model=model_name,
    api_base=api_base,
    api_key=api_key
)

class LineItem(BaseModel):
    """A line item in an invoice."""
    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""
    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )

pdf_reader = PDFReader()
documents = pdf_reader.load_data(file=Path("rag/data/pdf测试.pdf"))
text = documents[0].text

sllm = llm.as_structured_llm(Invoice)
response = sllm.complete(text)

json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))