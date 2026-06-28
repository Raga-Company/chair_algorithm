import logging
from typing import List, Dict, Any
from collections import Counter
from preprocessing.cleaner import clean_text

logger = logging.getLogger(__name__)

class PseudoRelevanceFeedback:
    """
    پیاده‌سازی بازخورد شبه‌ارتباطی (PRF) برای بهبود پرسش
    """
    
    def __init__(self, bm25_retriever, top_k=5, expansion_terms=5):
        """
        Args:
            bm25_retriever: نمونه از BM25Retriever
            top_k: تعداد اسناد اول برای استخراج کلمات
            expansion_terms: تعداد کلمات اضافه‌شده به پرسش
        """
        self.bm25 = bm25_retriever
        self.top_k = top_k
        self.expansion_terms = expansion_terms
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self):
        """بارگذاری stopwords فارسی"""
        return {
            'و', 'به', 'از', 'با', 'برای', 'در', 'را', 'که', 'چه', 'چی',
            'چگونه', 'چرا', 'کی', 'کجا', 'آیا', 'خواهم', 'خواهی', 'خواهد',
            'باشم', 'باشی', 'باشد', 'هستم', 'هستی', 'است', 'ایم', 'اید',
            'اند', 'شد', 'شو', 'شده', 'کرد', 'کن', 'کند', 'ده', 'دهد',
            'گیر', 'گیرد', 'آورد', 'آورده', 'مثل', 'مانند', 'بر', 'روی'
        }
    
    def extract_key_terms(self, documents: List[Dict]) -> List[str]:
        """
        استخراج کلمات کلیدی از اسناد
        
        Args:
            documents: لیست اسناد بازیابی‌شده
        
        Returns:
            لیست کلمات کلیدی به همراه وزن
        """
        term_freq = Counter()
        
        for doc in documents:
            # ترکیب فیلدهای متنی
            text = f"{doc.get('question', '')} {doc.get('answer', '')} {doc.get('category', '')} {doc.get('specialty', '')}"
            text = clean_text(text)
            tokens = text.split()
            
            # حذف stopwords
            tokens = [t for t in tokens if t not in self.stopwords and len(t) > 2]
            
            # وزن‌دهی: کلمات در سوال اهمیت بیشتری دارند
            question_tokens = clean_text(doc.get('question', '')).split()
            for token in tokens:
                weight = 2.0 if token in question_tokens else 1.0
                term_freq[token] += weight
        
        # بازگرداندن کلمات با بیشترین تکرار
        return [term for term, _ in term_freq.most_common(self.expansion_terms)]
    
    def expand_query(self, query: str) -> str:
        """
        گسترش پرسش با استفاده از بازخورد شبه‌ارتباطی
        
        Args:
            query: پرسش اصلی
        
        Returns:
            پرسش گسترش‌یافته
        """
        # بازیابی اولیه
        initial_results = self.bm25.search(query, k=self.top_k)
        
        if not initial_results:
            logger.warning("No initial results for PRF, returning original query")
            return query
        
        # استخراج کلمات کلیدی
        key_terms = self.extract_key_terms(initial_results)
        
        if not key_terms:
            return query
        
        # ترکیب با پرسش اصلی (افزودن کلمات جدید)
        expanded_query = query + " " + " ".join(key_terms)
        logger.info(f"PRF expanded: '{query}' -> '{expanded_query}'")
        
        return expanded_query
    
    def expand_with_weights(self, query: str, alpha: float = 0.7) -> str:
        """
        گسترش پرسش با وزن‌دهی به کلمات جدید
        
        Args:
            query: پرسش اصلی
            alpha: وزن پرسش اصلی (0-1)
        
        Returns:
            پرسش گسترش‌یافته با وزن‌دهی
        """
        initial_results = self.bm25.search(query, k=self.top_k)
        
        if not initial_results:
            return query
        
        # استخراج کلمات با وزن
        term_weights = {}
        for doc in initial_results:
            text = f"{doc.get('question', '')} {doc.get('answer', '')}"
            text = clean_text(text)
            tokens = text.split()
            for token in tokens:
                if token not in self.stopwords and len(token) > 2:
                    term_weights[token] = term_weights.get(token, 0) + 1
        
        # مرتب‌سازی بر اساس وزن
        sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
        top_terms = [term for term, _ in sorted_terms[:self.expansion_terms]]
        
        # ترکیب وزنی
        expanded = query
        for term in top_terms:
            if term not in query:
                expanded += f" {term}"
        
        return expanded


def get_prf_expander(bm25_retriever, top_k=5, expansion_terms=5):
    return PseudoRelevanceFeedback(bm25_retriever, top_k, expansion_terms)