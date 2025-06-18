#!/usr/bin/env python3
"""
Quality scoring system for synthetic data
Scores instruction-response pairs on multiple dimensions without using APIs
"""

import re
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter

import textstat
import nltk
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


@dataclass
class QualityDimensions:
    """Weights for different quality dimensions"""
    clarity: float = 0.25
    response_quality: float = 0.35
    diversity: float = 0.20
    complexity: float = 0.20


class QualityScorer:
    """
    Scores synthetic data quality without using external APIs
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        dimensions: Optional[QualityDimensions] = None,
        cache_scores: bool = True,
        cache_path: str = "quality_cache.pkl"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dimensions = dimensions or QualityDimensions()
        self.cache_scores = cache_scores
        self.cache_path = cache_path
        self.score_cache = self._load_cache()
        
        # Load instruction patterns for clarity scoring
        self.instruction_patterns = self._load_instruction_patterns()
        self.stop_words = set(stopwords.words('english'))
    
    def __del__(self):
        """Save cache upon object destruction."""
        self.save_cache()

    def _load_cache(self) -> Dict:
        """Load score cache from disk if it exists."""
        if not self.cache_scores or not os.path.exists(self.cache_path):
            return {}
        
        logger.info(f"Loading score cache from {self.cache_path}")
        try:
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Could not load cache file: {e}. Starting with an empty cache.")
            return {}

    def save_cache(self):
        """Save the current score cache to disk."""
        if not self.cache_scores or not self.score_cache:
            return
            
        logger.info(f"Saving score cache to {self.cache_path} with {len(self.score_cache)} items.")
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.score_cache, f)
        except Exception as e:
            logger.error(f"Failed to save quality score cache: {e}")
    
    def _load_instruction_patterns(self) -> List[str]:
        """Common patterns for clear instructions"""
        return [
            r'^(write|create|generate|compose)\s+',
            r'^(explain|describe|define|clarify)\s+',
            r'^(list|enumerate|name|provide)\s+',
            r'^(analyze|evaluate|compare|contrast)\s+',
            r'^(solve|calculate|compute|determine)\s+',
            r'^(translate|convert|transform|change)\s+',
            r'^(summarize|outline|brief|condense)\s+',
            r'^(what|why|how|when|where|who)\s+',
            r'(step[s]?\s+by\s+step|detailed|specific)',
            r'(example[s]?|instance[s]?|illustration[s]?)',
        ]
    
    def score_sample(self, sample: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Score a single instruction-response pair
        
        Args:
            sample: Dictionary with 'instruction' and 'response' keys
            
        Returns:
            Tuple of (total_score, component_scores)
        """
        # Check cache
        cache_key = f"{sample.get('instruction', '')}_{sample.get('response', '')}"
        if self.cache_scores and cache_key in self.score_cache:
            return self.score_cache[cache_key]
        
        instruction = sample.get('instruction', sample.get('prompt', ''))
        response = sample.get('response', sample.get('output', sample.get('completion', '')))
        
        # Score each dimension
        scores = {
            'clarity': self.score_clarity(instruction),
            'response_quality': self.score_response_quality(response, instruction),
            'diversity': self.score_diversity(instruction, response),
            'complexity': self.score_complexity(instruction, response)
        }
        
        # Calculate weighted total
        total_score = sum(
            scores[dim] * getattr(self.dimensions, dim)
            for dim in scores
        )
        
        # Cache result
        if self.cache_scores:
            # Check if cache has grown enough to save
            should_save = cache_key not in self.score_cache
            self.score_cache[cache_key] = (total_score, scores)
            # Periodically save to avoid data loss on interruption
            if should_save and len(self.score_cache) % 500 == 0:
                self.save_cache()
        
        return total_score, scores
    
    def score_clarity(self, instruction: str) -> float:
        """Score instruction clarity (0-1)"""
        if not instruction:
            return 0.0
        
        scores = []
        
        # 1. Length appropriateness
        word_count = len(instruction.split())
        if word_count < 3:
            length_score = 0.3
        elif word_count > 100:
            length_score = 0.7
        elif 5 <= word_count <= 50:
            length_score = 1.0
        else:
            length_score = 0.85
        scores.append(length_score)
        
        # 2. Structural clarity
        has_question = '?' in instruction
        has_clear_directive = any(
            re.search(pattern, instruction.lower())
            for pattern in self.instruction_patterns
        )
        
        if has_question or has_clear_directive:
            structure_score = 1.0
        else:
            structure_score = 0.6
        scores.append(structure_score)
        
        # 3. Readability
        try:
            # Flesch Reading Ease (higher is easier)
            readability = textstat.flesch_reading_ease(instruction)
            # Convert to 0-1 scale (30-80 is good range)
            if readability < 30:
                read_score = 0.6
            elif readability > 80:
                read_score = 0.9
            else:
                read_score = 0.7 + (readability - 30) / 200
        except:
            read_score = 0.7
        scores.append(read_score)
        
        # 4. Specificity (penalize vague instructions)
        vague_words = ['something', 'stuff', 'thing', 'whatever', 'somehow']
        vague_count = sum(1 for word in vague_words if word in instruction.lower())
        specificity_score = max(0.4, 1.0 - (vague_count * 0.2))
        scores.append(specificity_score)
        
        # 5. Grammar indicators (simple check)
        sentences = sent_tokenize(instruction)
        if sentences:
            # Check if sentences start with capital letters
            capital_ratio = sum(1 for s in sentences if s and s[0].isupper()) / len(sentences)
            grammar_score = 0.5 + (capital_ratio * 0.5)
        else:
            grammar_score = 0.6
        scores.append(grammar_score)
        
        return np.mean(scores)
    
    def score_response_quality(self, response: str, instruction: str = "") -> float:
        """Score response quality (0-1)"""
        if not response:
            return 0.0
        
        scores = []
        
        # 1. Length appropriateness
        word_count = len(response.split())
        if word_count < 5:
            length_score = 0.2
        elif word_count > 1000:
            length_score = 0.7
        elif 20 <= word_count <= 500:
            length_score = 1.0
        else:
            length_score = 0.8
        scores.append(length_score)
        
        # 2. Structure quality
        has_paragraphs = '\n\n' in response or '\n' in response
        has_list = bool(re.search(r'^\s*[\d•\-\*]\s*', response, re.MULTILINE))
        has_code = '```' in response or bool(re.search(r'def\s+\w+|function\s+\w+', response))
        
        structure_score = 0.6  # Base score
        if has_paragraphs:
            structure_score += 0.2
        if has_list:
            structure_score += 0.1
        if has_code and ('code' in instruction.lower() or 'function' in instruction.lower()):
            structure_score += 0.1
        scores.append(min(1.0, structure_score))
        
        # 3. Coherence (sentence flow)
        sentences = sent_tokenize(response)
        if len(sentences) > 2:
            # Check sentence length variation
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = np.std(sentence_lengths) / (np.mean(sentence_lengths) + 1)
            
            # Good variance is between 0.3 and 0.7
            if 0.3 <= length_variance <= 0.7:
                coherence_score = 1.0
            else:
                coherence_score = 0.7
        else:
            coherence_score = 0.8
        scores.append(coherence_score)
        
        # 4. Completeness (does it end properly?)
        response_stripped = response.strip()
        if response_stripped:
            # Check for proper ending
            ends_with_punctuation = response_stripped[-1] in '.!?'
            ends_mid_sentence = response_stripped.endswith((',', ';', 'and', 'or', 'the'))
            
            if ends_with_punctuation and not ends_mid_sentence:
                completeness_score = 1.0
            elif ends_mid_sentence:
                completeness_score = 0.4
            else:
                completeness_score = 0.7
        else:
            completeness_score = 0.0
        scores.append(completeness_score)
        
        # 5. Information density
        # Ratio of unique meaningful words to total words
        words = word_tokenize(response.lower())
        meaningful_words = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 2]
        
        if meaningful_words:
            unique_ratio = len(set(meaningful_words)) / len(meaningful_words)
            density_score = min(1.0, unique_ratio * 1.5)  # Scale up as unique is good
        else:
            density_score = 0.3
        scores.append(density_score)
        
        return np.mean(scores)
    
    def score_diversity(self, instruction: str, response: str) -> float:
        """Score diversity/uniqueness (0-1)"""
        combined = f"{instruction} {response}".lower()
        
        if not combined.strip():
            return 0.0
        
        scores = []
        
        # 1. Vocabulary diversity
        tokens = self.tokenizer.tokenize(combined)
        if tokens:
            unique_ratio = len(set(tokens)) / len(tokens)
            vocab_score = min(1.0, unique_ratio * 2)  # Scale up
        else:
            vocab_score = 0.0
        scores.append(vocab_score)
        
        # 2. N-gram diversity
        words = combined.split()
        if len(words) >= 2:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
            scores.append(bigram_diversity)
        
        if len(words) >= 3:
            trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
            trigram_diversity = len(set(trigrams)) / len(trigrams) if trigrams else 0
            scores.append(trigram_diversity)
        
        # 3. Topic diversity (penalize overused topics)
        common_topics = [
            'python', 'code', 'programming', 'function', 'def',
            'explain', 'write', 'create', 'list', 'what',
            'machine learning', 'ai', 'data', 'algorithm'
        ]
        
        topic_count = sum(1 for topic in common_topics if topic in combined)
        topic_penalty = min(topic_count * 0.05, 0.3)  # Max 30% penalty
        topic_score = 1.0 - topic_penalty
        scores.append(topic_score)
        
        # 4. Pattern diversity (check for templated responses)
        template_patterns = [
            r'^here (is|are)',
            r'^this (is|are)',
            r'^the \w+ (is|are)',
            r'^\d+\.',  # Numbered lists
            r'^step \d+:',
            r'^first,.*second,.*third',
        ]
        
        pattern_matches = sum(
            1 for pattern in template_patterns
            if re.search(pattern, response.lower(), re.MULTILINE)
        )
        pattern_score = max(0.5, 1.0 - (pattern_matches * 0.1))
        scores.append(pattern_score)
        
        return np.mean(scores)
    
    def score_complexity(self, instruction: str, response: str) -> float:
        """Score reasoning/knowledge complexity (0-1)"""
        scores = []
        
        # 1. Instruction complexity
        complex_indicators = [
            'analyze', 'evaluate', 'compare', 'contrast', 'synthesize',
            'critique', 'design', 'develop', 'formulate', 'investigate',
            'why', 'how', 'explain the relationship', 'what if'
        ]
        
        instruction_lower = instruction.lower()
        complexity_matches = sum(1 for ind in complex_indicators if ind in instruction_lower)
        instruction_complexity = min(1.0, 0.5 + (complexity_matches * 0.25))
        scores.append(instruction_complexity)
        
        # 2. Response depth (multi-step reasoning)
        step_indicators = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'step 1', 'step 2', 'additionally', 'furthermore', 'moreover',
            'because', 'therefore', 'thus', 'consequently', 'as a result'
        ]
        
        response_lower = response.lower()
        step_count = sum(1 for ind in step_indicators if ind in response_lower)
        reasoning_score = min(1.0, 0.4 + (step_count * 0.15))
        scores.append(reasoning_score)
        
        # 3. Technical content
        technical_indicators = [
            '```', 'def ', 'function', 'class ', 'import ',
            'algorithm', 'complexity', 'O(', 'equation', 'formula',
            'theorem', 'hypothesis', 'variable', 'parameter'
        ]
        
        technical_count = sum(1 for ind in technical_indicators if ind in response_lower)
        technical_score = min(1.0, 0.5 + (technical_count * 0.1))
        scores.append(technical_score)
        
        # 4. Knowledge integration (references to concepts)
        knowledge_indicators = [
            'research', 'studies', 'evidence', 'theory', 'principle',
            'law of', 'according to', 'based on', 'demonstrates', 'indicates'
        ]
        
        knowledge_count = sum(1 for ind in knowledge_indicators if ind in response_lower)
        knowledge_score = min(1.0, 0.6 + (knowledge_count * 0.2))
        scores.append(knowledge_score)
        
        # 5. Structural complexity
        # Longer, well-structured responses indicate complexity
        sentences = sent_tokenize(response)
        if len(sentences) > 3:
            # Check for subordinate clauses
            subordinate_indicators = ['which', 'that', 'who', 'whom', 'whose', 'where', 'when']
            clause_count = sum(1 for sent in sentences for ind in subordinate_indicators if ind in sent.lower())
            structure_score = min(1.0, 0.5 + (clause_count / len(sentences)) * 0.5)
        else:
            structure_score = 0.6
        scores.append(structure_score)
        
        return np.mean(scores)
    
    def score_batch(
        self,
        samples: List[Dict],
        show_progress: bool = True
    ) -> List[Tuple[float, Dict[str, float]]]:
        """Score a batch of samples"""
        
        results = []
        iterator = tqdm(samples, desc="Scoring samples") if show_progress else samples
        
        for sample in iterator:
            score, components = self.score_sample(sample)
            results.append((score, components))
        
        return results
    
    def get_statistics(self, scores: List[Tuple[float, Dict[str, float]]]) -> Dict:
        """Get statistics about score distribution"""
        
        total_scores = [s[0] for s in scores]
        component_scores = {
            'clarity': [s[1]['clarity'] for s in scores],
            'response_quality': [s[1]['response_quality'] for s in scores],
            'diversity': [s[1]['diversity'] for s in scores],
            'complexity': [s[1]['complexity'] for s in scores]
        }
        
        stats = {
            'total': {
                'mean': np.mean(total_scores),
                'std': np.std(total_scores),
                'min': np.min(total_scores),
                'max': np.max(total_scores),
                'median': np.median(total_scores),
                'percentiles': {
                    '25': np.percentile(total_scores, 25),
                    '50': np.percentile(total_scores, 50),
                    '75': np.percentile(total_scores, 75),
                    '90': np.percentile(total_scores, 90),
                    '95': np.percentile(total_scores, 95),
                    '99': np.percentile(total_scores, 99)
                }
            }
        }
        
        # Component statistics
        for component, values in component_scores.items():
            stats[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return stats


def test_quality_scorer():
    """Test the quality scorer with example data"""
    
    scorer = QualityScorer()
    
    # Test samples
    test_samples = [
        {
            "instruction": "Write a Python function to calculate factorial.",
            "response": """def factorial(n):
    if n < 0:
        return None
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result"""
        },
        {
            "instruction": "Explain photosynthesis.",
            "response": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This process occurs in the chloroplasts of plant cells and is essential for life on Earth."
        },
        {
            "instruction": "wut is 2+2",
            "response": "4"
        },
        {
            "instruction": "Analyze the causes of World War I and their interconnections.",
            "response": """World War I resulted from a complex web of interconnected causes:

First, the alliance system created two opposing blocks: the Triple Alliance (Germany, Austria-Hungary, Italy) and the Triple Entente (France, Russia, Britain). These alliances meant that a conflict between two nations could escalate into a continental war.

Second, nationalism fueled tensions, particularly in the Balkans where Slavic peoples sought independence from Austria-Hungary. The assassination of Archduke Franz Ferdinand by a Serbian nationalist provided the immediate trigger.

Third, imperialism created competition for colonies and resources, particularly between Germany and established powers like Britain and France. This rivalry extended to naval arms races and economic competition.

Finally, militarism meant that nations had large standing armies and detailed war plans, making military solutions seem viable. The Schlieffen Plan, for instance, required Germany to attack France through Belgium, which brought Britain into the war.

These factors created a powder keg that needed only a spark to explode into the devastating conflict of 1914-1918."""
        }
    ]
    
    # Score samples
    for i, sample in enumerate(test_samples):
        score, components = scorer.score_sample(sample)
        print(f"\nSample {i+1}:")
        print(f"Instruction: {sample['instruction'][:50]}...")
        print(f"Total Score: {score:.3f}")
        print(f"Components: {json.dumps(components, indent=2)}")


if __name__ == "__main__":
    test_quality_scorer() 