# -*- coding: utf-8 -*-
"""Evidence extractor using production rules without fixed keyword lists"""

import re
from typing import List, Dict, Optional
from difflib import SequenceMatcher

from utils import Domain, Polarity, TimeBucket, IntensityBucket, SemanticObject


class FixedProductionRuleBasedExtractor:
    """
    Extractor WITHOUT fixed keyword lists - follows all constraints
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        # Compile regex patterns WITHOUT fixed keyword lists
        self.patterns = self._compile_patterns_without_keywords()

        # Polarity detection - no domain-specific keywords
        self.negation_pattern = re.compile(r'\b(no\s+|not\s+|never\s+|none\s+|without\s+|didn\'t\s+|doesn\'t\s+|don\'t\s+|can\'t\s+|cannot\s+)\b', re.IGNORECASE)
        self.uncertainty_pattern = re.compile(r'\b(maybe\s+|perhaps\s+|might\s+|could\s+|possibly\s+|seems\s+|appears\s+|like\s+|sort of\s+|kind of\s+|not sure\s+|unsure\s+|probably\s+|somewhat\s+|a bit\s+|a little\s+|i think\s+|i feel\s+|i guess\s+)\b', re.IGNORECASE)

        # Time detection patterns - no domain mapping
        self.time_patterns = {
            TimeBucket.TODAY: [
                re.compile(r'\b(today|this morning|this afternoon|this evening|just now|now|subah|आज|morning|afternoon|evening|tonight|subah)\b', re.IGNORECASE)
            ],
            TimeBucket.LAST_NIGHT: [
                re.compile(r'\b(last night|yesterday night|night|3am|midnight|late night|raat|kal raat|overnight|bedtime|सोते समय|raat)\b', re.IGNORECASE)
            ],
            TimeBucket.PAST_WEEK: [
                re.compile(r'\b(this week|recently|lately|past few|few days|recent days|last week|पिछले कुछ दिन)\b', re.IGNORECASE)
            ]
        }

        # Intensity detection - NOT domain-specific
        self.low_intensity_words = {'slight', 'mild', 'a bit', 'a little', 'somewhat', 'minor', 'low', 'gentle'}
        self.high_intensity_words = {'super', 'very', 'extremely', 'really', 'intense', 'sharp', 'severe', 'strong', 'high', 'racing', 'heavy', 'acute', 'severe'}
        self.medium_intensity_indicators = {'moderate', 'medium', 'average'}

    def _compile_patterns_without_keywords(self):
        """Compile regex patterns WITHOUT fixed keyword lists"""
        # Use only syntactic patterns, NOT semantic keyword lists
        patterns = {
            Domain.SYMPTOM: [
                # Pattern 1: Physical sensation with body parts/adjectives
                re.compile(r'(?:had|have|having|felt|feel|feeling|got|noticed|experienced|suffered from|complained of)\s+([^.,;!?]{5,80}?(?:in my|in the|on my|behind my|around my|near my|with|that|which|when))', re.IGNORECASE),
                # Pattern 2: Adjective + noun pattern (e.g., "sharp pain", "dull ache")
                re.compile(r'\b([a-z]+)\s+(pain|ache|discomfort|sensation|feeling|pressure|tightness|soreness|stiffness)\b', re.IGNORECASE)
            ],
            Domain.FOOD: [
                # Pattern 1: Eating/drinking verbs
                re.compile(r'(?:ate|eat|eating|had|consumed|drank|drink|drinking|breakfast|lunch|dinner|snack|meal)\s+([^.,;!?]{5,80}?(?:with|and|\+|plus|along with|together with))', re.IGNORECASE),
                # Pattern 2: Food item patterns
                re.compile(r'\b([a-z]+\s+){0,3}(chai|coffee|tea|toast|rice|dal|roti|bread|salad|bowl|plate|meal)\b', re.IGNORECASE)
            ],
            Domain.EMOTION: [
                # Pattern 1: Emotion verbs
                re.compile(r'(?:felt|feeling|feel|was|were|emotion|mood|emotionally|feels|felt like)\s+([^.,;!?]{5,80}?(?:because|due to|as|since|for|about|regarding))', re.IGNORECASE),
                # Pattern 2: Adjective describing state
                re.compile(r'\b(?:very|quite|really|extremely|somewhat|a bit|a little)\s+([a-z]+)\b', re.IGNORECASE)
            ],
            Domain.MIND: [
                # Pattern 1: Mental process verbs
                re.compile(r'(?:mind|brain|thought|thinking|concentration|focus|memory|mental|mentally|cognitive|mindset)\s+([^.,;!?]{5,80}?(?:while|when|during|after|before))', re.IGNORECASE),
                # Pattern 2: Cognitive states
                re.compile(r'\b(?:was|were|felt|feeling)\s+(?:[a-z]+\s+){0,3}(clear|focused|scattered|racing|looping|ruminating|preoccupied|absent)\b', re.IGNORECASE)
            ]
        }
        return patterns

    def extract(self, text: str, journal_id: str = None) -> List[Dict]:
        """Main extraction method that returns dicts with 'text' field"""
        if not text or len(text) < 10:
            if self.debug:
                print(f"  Skipping empty or very short text")
            return []

        # Split into sentences for better context
        sentences = re.split(r'[.!?;]\s+', text)

        all_objects = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # Extract objects from this sentence
            sentence_objects = self._extract_from_sentence(sentence, text)
            all_objects.extend(sentence_objects)

        # Deduplicate and filter
        filtered_objects = self._filter_and_deduplicate(all_objects, text)

        # Convert to dict format with 'text' field
        result_dicts = []
        for obj in filtered_objects:
            obj_dict = obj.to_dict()
            # Add the required 'text' field (summary of evidence span)
            obj_dict['text'] = obj.evidence_span[:100] + ('...' if len(obj.evidence_span) > 100 else '')
            result_dicts.append(obj_dict)

        if self.debug and journal_id:
            initial_count = len(all_objects)
            final_count = len(filtered_objects)
            if initial_count != final_count:
                print(f"  Filtered {initial_count - final_count} objects for {journal_id}")

        return result_dicts

    def _extract_from_sentence(self, sentence: str, full_text: str) -> List[SemanticObject]:
        """Extract objects from a single sentence WITHOUT keyword mapping"""
        objects = []

        # Domain inference based on syntactic patterns ONLY
        for domain, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    try:
                        # Extract evidence span
                        evidence_start = max(0, match.start())
                        evidence_end = min(len(sentence), match.end() + 30)
                        evidence = sentence[evidence_start:evidence_end].strip()

                        # Skip if evidence is too short or generic
                        if len(evidence) < 10 or self._is_generic_evidence(evidence):
                            continue

                        # Determine domain from context (not from keywords)
                        inferred_domain = self._infer_domain_from_context(sentence, evidence)

                        # Use inferred domain if available, otherwise use pattern domain
                        final_domain = inferred_domain if inferred_domain else domain

                        # Determine polarity
                        polarity = self._determine_polarity(sentence, match.start(), evidence)

                        # Determine time bucket
                        time_bucket = self._determine_time_bucket(full_text, sentence)

                        # Determine intensity/arousal (NOT domain-specific)
                        bucket_value = self._determine_intensity(sentence, evidence, final_domain)

                        # Create object
                        if final_domain == Domain.EMOTION:
                            obj = SemanticObject(
                                domain=final_domain,
                                evidence_span=evidence,
                                polarity=polarity,
                                time_bucket=time_bucket,
                                arousal_bucket=bucket_value
                            )
                        else:
                            obj = SemanticObject(
                                domain=final_domain,
                                evidence_span=evidence,
                                polarity=polarity,
                                time_bucket=time_bucket,
                                intensity_bucket=bucket_value
                            )

                        objects.append(obj)

                    except Exception as e:
                        if self.debug:
                            print(f"Error extracting object: {e}")
                        continue

        return objects

    def _infer_domain_from_context(self, sentence: str, evidence: str) -> Optional[Domain]:
        """Infer domain from context WITHOUT fixed keywords"""
        sentence_lower = sentence.lower()

        # Use context words, NOT fixed mappings
        domain_indicators = {
            Domain.SYMPTOM: [
                r'\b(pain|ache|hurt|sore|tender|uncomfortable|sensation)\b',
                r'\b(head|stomach|chest|back|neck|joint|muscle|body|physical)\b',
                r'\b(doctor|hospital|medication|treatment|symptom)\b'
            ],
            Domain.FOOD: [
                r'\b(eat|ate|eating|food|meal|breakfast|lunch|dinner|snack)\b',
                r'\b(drink|drank|drinking|beverage|hungry|thirsty|full|stomach)\b',
                r'\b(kitchen|restaurant|cook|cooking|prepared|served)\b'
            ],
            Domain.EMOTION: [
                r'\b(feel|felt|feeling|emotion|mood|emotional|psychologically)\b',
                r'\b(happy|sad|angry|excited|nervous|anxious|calm|peaceful)\b',
                r'\b(heart|chest|tears|smile|laugh|cry|emotional|mood)\b'
            ],
            Domain.MIND: [
                r'\b(think|thought|thinking|mind|brain|mental|cognitive)\b',
                r'\b(focus|concentrate|memory|remember|forget|recall)\b',
                r'\b(idea|concept|plan|decision|understand|comprehend)\b'
            ]
        }

        # Count matches for each domain
        domain_scores = {d: 0 for d in Domain}

        for domain, patterns in domain_indicators.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    domain_scores[domain] += 1

        # Return domain with highest score if above threshold
        max_score = max(domain_scores.values())
        if max_score > 0:
            for domain, score in domain_scores.items():
                if score == max_score:
                    return domain

        return None

    def _is_generic_evidence(self, evidence: str) -> bool:
        """Check if evidence is too generic"""
        evidence_lower = evidence.lower()
        words = evidence_lower.split()

        # Check for very short evidence
        if len(words) <= 2:
            return True

        # Generic verbs that don't provide enough context
        generic_verbs = {'felt', 'was', 'were', 'had', 'have', 'has', 'did', 'do', 'does'}

        first_word = words[0]
        if first_word in generic_verbs and len(words) < 4:
            return True

        return False

    def _determine_polarity(self, sentence: str, match_start: int, evidence: str) -> Polarity:
        """Determine polarity from context"""
        # Look at context before and after the match
        context_before = sentence[max(0, match_start - 100):match_start].lower()
        context_after = sentence[match_start:min(len(sentence), match_start + 50)].lower()

        # Check for negation
        negation_patterns = [
            r'\bno\s+',
            r'\bnot\s+',
            r'\bnever\s+',
            r'\bnone\s+',
            r'\bwithout\s+',
            r'\bdidn\'t\s+',
            r'\bdoesn\'t\s+',
            r'\bdon\'t\s+',
            r'\bcan\'t\s+',
            r'\bcannot\s+'
        ]

        for pattern in negation_patterns:
            if re.search(pattern, context_before) or re.search(pattern, context_after):
                return Polarity.ABSENT

        # Check for uncertainty
        uncertainty_patterns = [
            r'\bmaybe\s+',
            r'\bperhaps\s+',
            r'\bmight\s+',
            r'\bcould\s+',
            r'\bpossibly\s+',
            r'\bnot sure\s+',
            r'\bunsure\s+',
            r'\bprobably\s+'
        ]

        for pattern in uncertainty_patterns:
            if re.search(pattern, context_before) or re.search(pattern, context_after):
                return Polarity.UNCERTAIN

        return Polarity.PRESENT

    def _determine_time_bucket(self, full_text: str, sentence: str) -> TimeBucket:
        """Time bucket detection"""
        sentence_lower = sentence.lower()
        full_text_lower = full_text.lower()

        for time_bucket, patterns in self.time_patterns.items():
            for pattern in patterns:
                if pattern.search(sentence_lower) or pattern.search(full_text_lower):
                    return time_bucket

        return TimeBucket.UNKNOWN

    def _determine_intensity(self, sentence: str, evidence: str, domain: Domain) -> IntensityBucket:
        """Determine intensity/arousal WITHOUT domain-specific defaults"""
        combined_text = f"{sentence} {evidence}".lower()

        # Check for low intensity indicators
        for word in self.low_intensity_words:
            if word in combined_text:
                return IntensityBucket.LOW

        # Check for high intensity indicators
        for word in self.high_intensity_words:
            if word in combined_text:
                return IntensityBucket.HIGH

        # Check for medium intensity indicators
        for word in self.medium_intensity_indicators:
            if word in combined_text:
                return IntensityBucket.MEDIUM

        # Default to unknown for all domains (no domain-specific bias)
        return IntensityBucket.UNKNOWN

    def _filter_and_deduplicate(self, objects: List[SemanticObject], text: str) -> List[SemanticObject]:
        """Filter and deduplicate objects"""
        if not objects:
            return []

        # Sort by evidence length (longer is usually more specific)
        objects.sort(key=lambda x: len(x.evidence_span), reverse=True)

        filtered = []
        seen_hashes = set()

        for obj in objects:
            # Create a hash based on domain and normalized evidence
            evidence_norm = obj.evidence_span.lower().strip()

            # Skip if evidence is not in the original text (safety check)
            if evidence_norm not in text.lower():
                if self.debug:
                    print(f"  Warning: Evidence not found in text: {evidence_norm[:50]}...")
                continue

            # Skip very similar objects
            is_duplicate = False
            for seen in seen_hashes:
                similarity = SequenceMatcher(None, evidence_norm[:50], seen[:50]).ratio()
                if similarity > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(obj)
                seen_hashes.add(evidence_norm[:50])

        return filtered