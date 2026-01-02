# -*- coding: utf-8 -*-
"""Utility classes and functions for evidence-grounded extraction"""

import json
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    SYMPTOM = "symptom"
    FOOD = "food"
    EMOTION = "emotion"
    MIND = "mind"


class Polarity(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class TimeBucket(str, Enum):
    TODAY = "today"
    LAST_NIGHT = "last_night"
    PAST_WEEK = "past_week"
    UNKNOWN = "unknown"


class IntensityBucket(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass
class SemanticObject:
    """Structured extraction from journal text"""
    domain: Domain
    evidence_span: str
    polarity: Polarity
    time_bucket: TimeBucket
    intensity_bucket: Optional[IntensityBucket] = None
    arousal_bucket: Optional[IntensityBucket] = None

    def to_dict(self) -> Dict:
        result = {
            "domain": self.domain.value,
            "evidence_span": self.evidence_span,
            "polarity": self.polarity.value,
            "time_bucket": self.time_bucket.value
        }
        if self.domain == Domain.EMOTION:
            result["arousal_bucket"] = self.arousal_bucket.value if self.arousal_bucket else "unknown"
        else:
            result["intensity_bucket"] = self.intensity_bucket.value if self.intensity_bucket else "unknown"
        return result

    @classmethod
    def from_dict(cls, data: Dict):
        """Create SemanticObject from dictionary"""
        domain = Domain(data['domain'])
        polarity = Polarity(data['polarity'])
        time_bucket = TimeBucket(data['time_bucket'])

        if domain == Domain.EMOTION:
            return cls(
                domain=domain,
                evidence_span=data['evidence_span'],
                polarity=polarity,
                time_bucket=time_bucket,
                arousal_bucket=IntensityBucket(data.get('arousal_bucket', 'unknown'))
            )
        else:
            return cls(
                domain=domain,
                evidence_span=data['evidence_span'],
                polarity=polarity,
                time_bucket=time_bucket,
                intensity_bucket=IntensityBucket(data.get('intensity_bucket', 'unknown'))
            )


class DataLoader:
    """Handle all data loading operations with robust error handling"""

    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict]:
        """
        Load JSONL file with multiple fallback strategies
        """
        data = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Strategy 1: Try direct JSON parse
                try:
                    parsed = json.loads(line)
                    data.append(parsed)
                    continue
                except json.JSONDecodeError:
                    pass

                # Strategy 2: Try Python literal_eval (for single quotes)
                try:
                    # Clean common issues
                    cleaned = line
                    if cleaned.endswith('...'):
                        cleaned = cleaned[:-3]

                    # Fix trailing commas
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*]', ']', cleaned)

                    parsed = ast.literal_eval(cleaned)
                    data.append(parsed)
                    continue
                except:
                    pass

                # Strategy 3: Manual parsing for common patterns
                try:
                    parsed = DataLoader._manual_parse(line)
                    if parsed:
                        data.append(parsed)
                        continue
                except:
                    pass

                print(f"Warning: Skipped line {i} in {filepath}")

        return data

    @staticmethod
    def _manual_parse(line: str) -> Optional[Dict]:
        """Manual parsing for specific patterns"""
        # Pattern 1: Journal entries with journal_id and text
        journal_pattern = r"'journal_id':\s*'([^']+)',\s*'created_at':\s*'([^']+)',\s*'text':\s*'([^']+)'"
        match = re.search(journal_pattern, line)
        if match:
            return {
                'journal_id': match.group(1),
                'created_at': match.group(2),
                'text': match.group(3)
            }

        # Pattern 2: Gold entries
        gold_pattern = r"'journal_id':\s*'([^']+)',\s*'items':\s*(\[[^\]]+\])"
        match = re.search(gold_pattern, line)
        if match:
            try:
                items = ast.literal_eval(match.group(2))
                return {
                    'journal_id': match.group(1),
                    'items': items
                }
            except:
                pass

        return None

    @staticmethod
    def load_gold_objects(filepath: str) -> Dict[str, List[SemanticObject]]:
        """Load gold objects from file"""
        gold_data = DataLoader.load_jsonl(filepath)
        gold_dict = {}

        for entry in gold_data:
            journal_id = entry.get('journal_id')
            if not journal_id:
                continue

            objects = []
            for item in entry.get('items', []):
                try:
                    obj = SemanticObject.from_dict(item)
                    objects.append(obj)
                except Exception as e:
                    print(f"Error loading gold object in {journal_id}: {e}")
                    continue

            gold_dict[journal_id] = objects

        return gold_dict

    @staticmethod
    def load_journals(filepath: str) -> Dict[str, str]:
        """Load journals into dictionary"""
        journals_data = DataLoader.load_jsonl(filepath)
        journals_dict = {}

        for entry in journals_data:
            journal_id = entry.get('journal_id')
            text = entry.get('text', '')
            if journal_id and text:
                journals_dict[journal_id] = text

        return journals_dict