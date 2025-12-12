import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Entity:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    source_text: str
    confidence: float


class EntityExtractor:
    def __init__(self):
        self.date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b',
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        ]
        
        self.id_patterns = [
            r'\b(GR[-_]?\d+)\b',
            r'\b(CASE[-_]?\d+)\b',
            r'\b(EMP[-_]?\d+)\b',
            r'\b(ID[-_:]?\s*\d+)\b',
            r'\b(#\d+)\b',
            r'\b(REF[-_]?\d+)\b',
        ]
        
        self.status_keywords = [
            'pending', 'approved', 'rejected', 'resolved', 'open', 'closed',
            'in progress', 'under review', 'escalated', 'completed', 'cancelled',
            'submitted', 'acknowledged', 'processing', 'finalized'
        ]
        
        self.department_keywords = [
            'hr', 'human resources', 'finance', 'it', 'information technology',
            'admin', 'administration', 'legal', 'operations', 'marketing',
            'sales', 'engineering', 'support', 'management', 'accounts',
            'procurement', 'logistics', 'quality', 'research', 'development'
        ]
        
        self.title_patterns = [
            r'\b(Mr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b(Mrs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b(Ms\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        ]
        
        self.money_patterns = [
            r'\b(Rs\.?\s*[\d,]+(?:\.\d{2})?)\b',
            r'\b(INR\s*[\d,]+(?:\.\d{2})?)\b',
            r'\b(\$[\d,]+(?:\.\d{2})?)\b',
            r'\b(₹[\d,]+(?:\.\d{2})?)\b',
        ]

    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(1),
                    entity_type='DATE',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        for pattern in self.id_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(1).upper(),
                    entity_type='ID',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))
        
        for keyword in self.status_keywords:
            pattern = r'\b(' + re.escape(keyword) + r')\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(1).lower(),
                    entity_type='STATUS',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
        
        for keyword in self.department_keywords:
            pattern = r'\b(' + re.escape(keyword) + r'(?:\s+department)?)\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(1).title(),
                    entity_type='DEPARTMENT',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
        
        for pattern in self.title_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    entity_type='PERSON',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        for pattern in self.money_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    entity_type='MONEY',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        entities = self._deduplicate_entities(entities)
        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        if not entities:
            return []
        
        entities.sort(key=lambda e: (e.start, -e.confidence))
        result = []
        
        for entity in entities:
            overlap = False
            for existing in result:
                if not (entity.end <= existing.start or entity.start >= existing.end):
                    overlap = True
                    break
            if not overlap:
                result.append(entity)
        
        return result

    def extract_facts(self, text: str, entities: List[Entity]) -> List[Fact]:
        facts = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence_entities = [e for e in entities if e.text.lower() in sentence.lower()]
            
            persons = [e for e in sentence_entities if e.entity_type == 'PERSON']
            statuses = [e for e in sentence_entities if e.entity_type == 'STATUS']
            dates = [e for e in sentence_entities if e.entity_type == 'DATE']
            ids = [e for e in sentence_entities if e.entity_type == 'ID']
            depts = [e for e in sentence_entities if e.entity_type == 'DEPARTMENT']
            
            for person in persons:
                for status in statuses:
                    facts.append(Fact(
                        subject=person.text,
                        predicate='has_status',
                        object=status.text,
                        source_text=sentence.strip(),
                        confidence=0.7
                    ))
            
            for id_ent in ids:
                for status in statuses:
                    facts.append(Fact(
                        subject=id_ent.text,
                        predicate='has_status',
                        object=status.text,
                        source_text=sentence.strip(),
                        confidence=0.85
                    ))
                for date in dates:
                    facts.append(Fact(
                        subject=id_ent.text,
                        predicate='dated',
                        object=date.text,
                        source_text=sentence.strip(),
                        confidence=0.8
                    ))
            
            for person in persons:
                for dept in depts:
                    facts.append(Fact(
                        subject=person.text,
                        predicate='belongs_to',
                        object=dept.text,
                        source_text=sentence.strip(),
                        confidence=0.6
                    ))
        
        return facts

    def create_entity_summary(self, entities: List[Entity], facts: List[Fact]) -> Dict:
        summary = {
            'entity_counts': {},
            'entities_by_type': {},
            'fact_count': len(facts),
            'key_relationships': []
        }
        
        for entity in entities:
            etype = entity.entity_type
            summary['entity_counts'][etype] = summary['entity_counts'].get(etype, 0) + 1
            if etype not in summary['entities_by_type']:
                summary['entities_by_type'][etype] = []
            if entity.text not in summary['entities_by_type'][etype]:
                summary['entities_by_type'][etype].append(entity.text)
        
        for fact in facts[:10]:
            summary['key_relationships'].append(
                f"{fact.subject} --[{fact.predicate}]--> {fact.object}"
            )
        
        return summary
