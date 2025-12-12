import re
from typing import Tuple


class ComplexityAnalyzer:
    def __init__(self):
        self.complex_keywords = [
            'compare', 'contrast', 'analyze', 'explain why', 'how does',
            'what are the differences', 'evaluate', 'assess', 'recommend',
            'suggest', 'implications', 'consequences', 'relationship between',
            'pros and cons', 'advantages and disadvantages', 'summarize',
            'elaborate', 'describe in detail', 'step by step', 'multiple',
            'all the', 'list all', 'comprehensive', 'detailed explanation'
        ]
        
        self.simple_keywords = [
            'what is', 'when is', 'where is', 'who is', 'how many',
            'how much', 'is there', 'are there', 'can i', 'do i',
            'yes or no', 'true or false', 'name of', 'date of'
        ]
    
    def count_question_parts(self, question: str) -> int:
        parts = re.split(r'[,;]|\band\b|\bor\b|\balso\b', question.lower())
        question_marks = question.count('?')
        return max(len([p for p in parts if len(p.strip()) > 10]), question_marks, 1)
    
    def has_complex_keywords(self, question: str) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.complex_keywords)
    
    def has_simple_keywords(self, question: str) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.simple_keywords)
    
    def calculate_complexity_score(self, question: str) -> float:
        score = 0.0
        
        word_count = len(question.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.1
        
        parts = self.count_question_parts(question)
        if parts > 2:
            score += 0.3
        elif parts > 1:
            score += 0.15
        
        if self.has_complex_keywords(question):
            score += 0.4
        
        if self.has_simple_keywords(question):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def analyze(self, question: str) -> Tuple[str, float, str]:
        complexity_score = self.calculate_complexity_score(question)
        
        if complexity_score >= 0.4:
            model_type = "LLM"
            reason = "Complex question requiring detailed analysis"
        else:
            model_type = "SLM"
            reason = "Simple factual question"
        
        return model_type, complexity_score, reason
