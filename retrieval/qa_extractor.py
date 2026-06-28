import logging
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

class PersianQAE:
    """
    استخراج پاسخ از متن با استفاده از مدل QA فارسی
    """
    def __init__(self, model_name="saman2000/bert-base-fa-qa"):
        """
        Args:
            model_name: نام مدل QA فارسی (پیش‌فرض: saman2000/bert-base-fa-qa)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.is_loaded = False
        
        try:
            logger.info(f"Loading QA model: {model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"✅ QA model loaded: {model_name} on {self.device}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load QA model '{model_name}': {e}")
            logger.warning("Trying fallback model: HooshvareLab/bert-fa-base-uncased")
            try:
                # Fallback: استفاده از مدل پایه (بدون fine-tuning) - ممکن است عملکرد ضعیف‌تری داشته باشد
                self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
                self.model = AutoModelForQuestionAnswering.from_pretrained("HooshvareLab/bert-fa-base-uncased")
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                logger.info("✅ QA model loaded with fallback: HooshvareLab/bert-fa-base-uncased")
            except Exception as e2:
                logger.error(f"❌ All QA models failed to load. QA will be disabled.")
                self.is_loaded = False
    
    def extract_answer(self, question: str, context: str, 
                      max_length: int = 512, 
                      min_answer_len: int = 2) -> Tuple[Optional[str], float]:
        """
        استخراج پاسخ از متن
        
        Args:
            question: سوال کاربر
            context: متن زمینه (سند)
            max_length: حداکثر طول توکن‌ها
            min_answer_len: حداقل طول پاسخ بر حسب کاراکتر
        
        Returns:
            tuple: (پاسخ استخراج‌شده، امتیاز اطمینان)
        """
        if not self.is_loaded:
            return None, 0.0
        
        if not question or not context:
            return None, 0.0
        
        try:
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
            
            start_scores = torch.softmax(start_logits, dim=1)
            end_scores = torch.softmax(end_logits, dim=1)
            
            start_idx = torch.argmax(start_scores, dim=1).item()
            end_idx = torch.argmax(end_scores, dim=1).item()
            
            confidence = (start_scores[0, start_idx] * end_scores[0, end_idx]).item()
            
            if start_idx >= end_idx or (end_idx - start_idx) < 1:
                return None, 0.0
            
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            if len(answer.strip()) < min_answer_len:
                return None, 0.0
            
            return answer.strip(), confidence
            
        except Exception as e:
            logger.warning(f"QA extraction failed: {e}")
            return None, 0.0
    
    def extract_batch(self, question: str, contexts: list, 
                     top_k: int = 3) -> list:
        """
        استخراج پاسخ از چند متن و بازگرداندن بهترین نتایج
        
        Args:
            question: سوال
            contexts: لیست متون
            top_k: تعداد نتایج برتر
        
        Returns:
            list: لیست پاسخ‌ها با امتیاز
        """
        if not self.is_loaded:
            return []
        
        results = []
        for i, context in enumerate(contexts):
            answer, confidence = self.extract_answer(question, context)
            if answer:
                results.append({
                    "index": i,
                    "answer": answer,
                    "confidence": confidence,
                    "context": context[:200] + "..." if len(context) > 200 else context
                })
        
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]


# نمونه برای استفاده سریع - با مدیریت خطا
try:
    default_qa = PersianQAE()
except Exception as e:
    logger.error(f"Could not initialize default QA: {e}")
    default_qa = None