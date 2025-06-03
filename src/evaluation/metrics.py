# src/evaluation/metrics.py
import re
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter, defaultdict
import logging

# Try to import optional dependencies
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. BLEU scores will use basic tokenization.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not available. ROUGE scores will be estimated.")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("bert-score not available. BERTScore will be skipped.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Semantic similarity will be skipped.")

logger = logging.getLogger(__name__)

def basic_tokenize(text: str) -> List[str]:
    """Basic tokenization for cases where NLTK is not available."""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def safe_tokenize(text: str) -> List[str]:
    """Safe tokenization with fallback."""
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text.lower())
        except LookupError:
            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                return word_tokenize(text.lower())
            except:
                pass
    
    return basic_tokenize(text)

def calculate_bleu_score(
    predictions: List[str], 
    references: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Calculate BLEU scores for predictions against references.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        max_n: Maximum n-gram order to calculate
    
    Returns:
        Dictionary with BLEU-1 through BLEU-4 scores
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    bleu_scores = {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)}
    
    if not predictions:
        return bleu_scores
    
    if NLTK_AVAILABLE:
        smoothing = SmoothingFunction().method1
        
        for i in range(1, max_n + 1):
            scores = []
            weights = [1.0/i] * i + [0.0] * (4 - i)  # Uniform weights up to n-gram
            
            for pred, ref in zip(predictions, references):
                pred_tokens = safe_tokenize(pred)
                ref_tokens = [safe_tokenize(ref)]  # BLEU expects list of reference lists
                
                if len(pred_tokens) == 0:
                    scores.append(0.0)
                else:
                    score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothing)
                    scores.append(score)
            
            bleu_scores[f'bleu_{i}'] = np.mean(scores)
    else:
        # Fallback implementation for basic BLEU
        for pred, ref in zip(predictions, references):
            pred_tokens = basic_tokenize(pred)
            ref_tokens = basic_tokenize(ref)
            
            # Calculate n-gram matches
            for n in range(1, max_n + 1):
                if len(pred_tokens) >= n and len(ref_tokens) >= n:
                    pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
                    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
                    
                    if pred_ngrams:
                        matches = sum(1 for ngram in pred_ngrams if ngram in ref_ngrams)
                        bleu_scores[f'bleu_{n}'] += matches / len(pred_ngrams)
        
        # Average across all samples
        for key in bleu_scores:
            bleu_scores[key] /= len(predictions)
    
    return bleu_scores

def calculate_rouge_scores(
    predictions: List[str], 
    references: List[str],
    rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']
) -> Dict[str, float]:
    """
    Calculate ROUGE scores for predictions against references.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        rouge_types: Types of ROUGE to calculate
    
    Returns:
        Dictionary with ROUGE scores
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    rouge_scores = {}
    
    if not predictions:
        return {rouge_type: 0.0 for rouge_type in rouge_types}
    
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        
        all_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            
            for rouge_type in rouge_types:
                # Use F1 score for ROUGE
                all_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        # Average scores
        for rouge_type in rouge_types:
            rouge_scores[rouge_type.lower()] = np.mean(all_scores[rouge_type])
    else:
        # Fallback implementation
        for rouge_type in rouge_types:
            if rouge_type.lower() == 'rouge1':
                rouge_scores['rouge1'] = _calculate_rouge_n(predictions, references, 1)
            elif rouge_type.lower() == 'rouge2':
                rouge_scores['rouge2'] = _calculate_rouge_n(predictions, references, 2)
            elif rouge_type.lower() == 'rougel':
                rouge_scores['rougel'] = _calculate_rouge_l(predictions, references)
    
    return rouge_scores

def _calculate_rouge_n(predictions: List[str], references: List[str], n: int) -> float:
    """Fallback ROUGE-N calculation."""
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = basic_tokenize(pred)
        ref_tokens = basic_tokenize(ref)
        
        if len(ref_tokens) < n:
            scores.append(0.0)
            continue
        
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        
        if not ref_ngrams:
            scores.append(0.0)
            continue
        
        matches = sum(1 for ngram in pred_ngrams if ngram in ref_ngrams)
        recall = matches / len(ref_ngrams)
        scores.append(recall)
    
    return np.mean(scores)

def _calculate_rouge_l(predictions: List[str], references: List[str]) -> float:
    """Fallback ROUGE-L calculation using LCS."""
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = basic_tokenize(pred)
        ref_tokens = basic_tokenize(ref)
        
        if not ref_tokens:
            scores.append(0.0)
            continue
        
        lcs_length = _longest_common_subsequence(pred_tokens, ref_tokens)
        
        if len(ref_tokens) == 0:
            scores.append(0.0)
        else:
            recall = lcs_length / len(ref_tokens)
            scores.append(recall)
    
    return np.mean(scores)

def _longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Calculate longest common subsequence length."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def calculate_bertscore(
    predictions: List[str], 
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en"
) -> Dict[str, float]:
    """
    Calculate BERTScore for predictions against references.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_type: BERT model to use for scoring
        lang: Language code
    
    Returns:
        Dictionary with precision, recall, and F1 BERTScores
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore not available, returning zero scores")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    try:
        P, R, F1 = bert_score(predictions, references, model_type=model_type, lang=lang, verbose=False)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    except Exception as e:
        logger.warning(f"BERTScore calculation failed: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

def calculate_semantic_similarity(
    predictions: List[str], 
    references: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, float]:
    """
    Calculate semantic similarity using sentence transformers.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_name: Sentence transformer model name
    
    Returns:
        Dictionary with similarity metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("Sentence transformers not available, returning zero scores")
        return {'cosine_similarity': 0.0}
    
    if not predictions:
        return {'cosine_similarity': 0.0}
    
    try:
        model = SentenceTransformer(model_name)
        
        # Encode texts
        pred_embeddings = model.encode(predictions)
        ref_embeddings = model.encode(references)
        
        # Calculate cosine similarities
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            similarity = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
            similarities.append(similarity)
        
        return {
            'cosine_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'std_similarity': np.std(similarities)
        }
    except Exception as e:
        logger.warning(f"Semantic similarity calculation failed: {e}")
        return {'cosine_similarity': 0.0}

def calculate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    stride: int = 256
) -> float:
    """
    Calculate perplexity of texts using the model.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of texts to evaluate
        max_length: Maximum sequence length
        stride: Sliding window stride
    
    Returns:
        Average perplexity
    """
    if not texts:
        return float('inf')
    
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(device)
            
            # Calculate loss
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def calculate_diversity_metrics(predictions: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated texts.
    
    Args:
        predictions: List of predicted texts
    
    Returns:
        Dictionary with diversity metrics
    """
    if not predictions:
        return {
            'unique_unigrams': 0.0,
            'unique_bigrams': 0.0,
            'unique_trigrams': 0.0,
            'repetition_rate': 0.0,
            'length_variance': 0.0
        }
    
    all_unigrams = []
    all_bigrams = []
    all_trigrams = []
    lengths = []
    
    for pred in predictions:
        tokens = basic_tokenize(pred)
        lengths.append(len(tokens))
        
        # Collect n-grams
        all_unigrams.extend(tokens)
        
        if len(tokens) >= 2:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            all_bigrams.extend(bigrams)
        
        if len(tokens) >= 3:
            trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens) - 2)]
            all_trigrams.extend(trigrams)
    
    # Calculate diversity
    unique_unigrams = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0
    unique_bigrams = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
    unique_trigrams = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0
    
    # Calculate repetition rate (proportion of repeated responses)
    unique_predictions = len(set(predictions))
    repetition_rate = 1 - (unique_predictions / len(predictions))
    
    # Length variance
    length_variance = np.var(lengths) if len(lengths) > 1 else 0
    
    return {
        'unique_unigrams': unique_unigrams,
        'unique_bigrams': unique_bigrams,
        'unique_trigrams': unique_trigrams,
        'repetition_rate': repetition_rate,
        'length_variance': length_variance,
        'avg_length': np.mean(lengths),
        'unique_responses_ratio': unique_predictions / len(predictions)
    }

def calculate_corporate_qa_metrics(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate corporate Q&A specific metrics.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with corporate-specific metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if not predictions:
        return {}
    
    # Corporate-specific keywords and patterns
    corporate_keywords = [
        'azure', 'ml', 'workspace', 'compute', 'endpoint', 'deployment',
        'project', 'request', 'service', 'portal', 'application', 'process',
        'policy', 'procedure', 'department', 'team', 'manager', 'employee',
        'staffing', 'fte', 'cwr', 'job code', 'so', 'pm', 'fc'
    ]
    
    action_words = [
        'create', 'setup', 'configure', 'deploy', 'submit', 'request',
        'apply', 'register', 'login', 'access', 'navigate', 'click'
    ]
    
    metrics = {}
    
    # Keyword coverage
    pred_keyword_coverage = []
    ref_keyword_coverage = []
    
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        pred_keywords = sum(1 for keyword in corporate_keywords if keyword in pred_lower)
        ref_keywords = sum(1 for keyword in corporate_keywords if keyword in ref_lower)
        
        pred_keyword_coverage.append(pred_keywords)
        ref_keyword_coverage.append(ref_keywords)
    
    metrics['avg_keywords_in_prediction'] = np.mean(pred_keyword_coverage)
    metrics['avg_keywords_in_reference'] = np.mean(ref_keyword_coverage)
    metrics['keyword_coverage_ratio'] = (
        np.mean(pred_keyword_coverage) / np.mean(ref_keyword_coverage) 
        if np.mean(ref_keyword_coverage) > 0 else 0
    )
    
    # Action word usage
    pred_actions = []
    ref_actions = []
    
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        pred_action_count = sum(1 for action in action_words if action in pred_lower)
        ref_action_count = sum(1 for action in action_words if action in ref_lower)
        
        pred_actions.append(pred_action_count)
        ref_actions.append(ref_action_count)
    
    metrics['avg_action_words_prediction'] = np.mean(pred_actions)
    metrics['avg_action_words_reference'] = np.mean(ref_actions)
    
    # Step-by-step structure detection
    step_indicators = ['step', 'first', 'second', 'third', 'next', 'then', 'finally', '1.', '2.', '3.']
    
    structured_predictions = 0
    structured_references = 0
    
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        if any(indicator in pred_lower for indicator in step_indicators):
            structured_predictions += 1
        
        if any(indicator in ref_lower for indicator in step_indicators):
            structured_references += 1
    
    metrics['structured_response_rate'] = structured_predictions / len(predictions)
    metrics['reference_structured_rate'] = structured_references / len(references)
    
    # Question answering completeness (basic heuristic)
    completeness_scores = []
    
    for pred, ref in zip(predictions, references):
        # Simple heuristic: longer responses that cover similar topics are more complete
        pred_words = set(basic_tokenize(pred))
        ref_words = set(basic_tokenize(ref))
        
        if not ref_words:
            completeness_scores.append(0.0)
        else:
            overlap = len(pred_words.intersection(ref_words))
            completeness = overlap / len(ref_words)
            completeness_scores.append(min(completeness, 1.0))
    
    metrics['avg_completeness'] = np.mean(completeness_scores)
    
    # Helpfulness indicators
    helpful_phrases = [
        'you can', 'to do this', 'follow these', 'here are the steps',
        'you need to', 'make sure to', 'remember to', 'it\'s important'
    ]
    
    helpful_predictions = 0
    for pred in predictions:
        pred_lower = pred.lower()
        if any(phrase in pred_lower for phrase in helpful_phrases):
            helpful_predictions += 1
    
    metrics['helpfulness_rate'] = helpful_predictions / len(predictions)
    
    return metrics

def calculate_readability_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate readability metrics for texts.
    
    Args:
        texts: List of texts to analyze
    
    Returns:
        Dictionary with readability metrics
    """
    if not texts:
        return {}
    
    # Basic readability metrics
    sentence_counts = []
    word_counts = []
    avg_word_lengths = []
    
    for text in texts:
        # Count sentences (basic approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_counts.append(len(sentences))
        
        # Count words
        words = basic_tokenize(text)
        word_counts.append(len(words))
        
        # Average word length
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            avg_word_lengths.append(avg_word_length)
        else:
            avg_word_lengths.append(0)
    
    return {
        'avg_sentences_per_response': np.mean(sentence_counts),
        'avg_words_per_response': np.mean(word_counts),
        'avg_word_length': np.mean(avg_word_lengths),
        'avg_words_per_sentence': (
            np.mean(word_counts) / np.mean(sentence_counts) 
            if np.mean(sentence_counts) > 0 else 0
        )
    }

def calculate_factuality_metrics(
    predictions: List[str], 
    references: List[str],
    factual_keywords: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate factuality-related metrics (basic heuristics).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        factual_keywords: Keywords that indicate factual content
    
    Returns:
        Dictionary with factuality metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if factual_keywords is None:
        factual_keywords = [
            'portal', 'website', 'url', 'link', 'page', 'menu', 'button',
            'form', 'field', 'option', 'setting', 'configuration', 'azure',
            'workspace', 'subscription', 'resource group', 'endpoint'
        ]
    
    # Factual keyword preservation
    keyword_preservation_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        ref_keywords = [kw for kw in factual_keywords if kw in ref_lower]
        
        if not ref_keywords:
            keyword_preservation_scores.append(1.0)  # No keywords to preserve
        else:
            preserved = sum(1 for kw in ref_keywords if kw in pred_lower)
            preservation_rate = preserved / len(ref_keywords)
            keyword_preservation_scores.append(preservation_rate)
    
    # Confidence indicators (words that suggest certainty vs uncertainty)
    confident_words = ['will', 'is', 'are', 'must', 'should', 'can', 'always']
    uncertain_words = ['might', 'may', 'could', 'possibly', 'perhaps', 'maybe']
    
    confidence_scores = []
    
    for pred in predictions:
        pred_lower = pred.lower()
        pred_words = basic_tokenize(pred_lower)
        
        confident_count = sum(1 for word in pred_words if word in confident_words)
        uncertain_count = sum(1 for word in pred_words if word in uncertain_words)
        
        total_indicator_words = confident_count + uncertain_count
        
        if total_indicator_words > 0:
            confidence_score = confident_count / total_indicator_words
        else:
            confidence_score = 0.5  # Neutral
        
        confidence_scores.append(confidence_score)
    
    return {
        'keyword_preservation': np.mean(keyword_preservation_scores),
        'avg_confidence_score': np.mean(confidence_scores),
        'high_confidence_responses': sum(1 for score in confidence_scores if score > 0.7) / len(confidence_scores)
    }

# Main evaluation function that combines all metrics
def evaluate_predictions_comprehensive(
    predictions: List[str],
    references: List[str],
    model=None,
    tokenizer=None,
    include_bertscore: bool = True,
    include_semantic: bool = True,
    include_perplexity: bool = True
) -> Dict[str, Dict]:
    """
    Run comprehensive evaluation with all available metrics.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model: Language model for perplexity calculation
        tokenizer: Tokenizer for perplexity calculation
        include_bertscore: Whether to calculate BERTScore
        include_semantic: Whether to calculate semantic similarity
        include_perplexity: Whether to calculate perplexity
    
    Returns:
        Dictionary with all calculated metrics
    """
    results = {}
    
    # Basic metrics
    results['bleu'] = calculate_bleu_score(predictions, references)
    results['rouge'] = calculate_rouge_scores(predictions, references)
    results['diversity'] = calculate_diversity_metrics(predictions)
    results['corporate_qa'] = calculate_corporate_qa_metrics(predictions, references)
    results['readability'] = calculate_readability_metrics(predictions)
    results['factuality'] = calculate_factuality_metrics(predictions, references)
    
    # Optional metrics that require additional dependencies
    if include_bertscore:
        results['bertscore'] = calculate_bertscore(predictions, references)
    
    if include_semantic:
        results['semantic_similarity'] = calculate_semantic_similarity(predictions, references)
    
    if include_perplexity and model is not None and tokenizer is not None:
        results['perplexity'] = calculate_perplexity(model, tokenizer, predictions)
    
    return results

if __name__ == "__main__":
    # Test the metrics functions
    predictions = [
        "To create a new project in Azure ML, go to the Azure portal and navigate to Machine Learning workspaces.",
        "You can create a compute cluster by clicking on the Compute tab in your workspace.",
        "The deployment process involves creating an endpoint and configuring the compute resources."
    ]
    
    references = [
        "To create a new project in Azure ML workspace, navigate to the Azure portal, select Machine Learning, and create a new workspace.",
        "Create a compute cluster by going to the Compute section in your Azure ML workspace and clicking Create.",
        "Deploy your model by creating a managed endpoint and setting up the appropriate compute instance."
    ]
    
    # Test basic metrics
    print("Testing metrics calculation...")
    
    bleu_scores = calculate_bleu_score(predictions, references)
    print(f"BLEU scores: {bleu_scores}")
    
    rouge_scores = calculate_rouge_scores(predictions, references)
    print(f"ROUGE scores: {rouge_scores}")
    
    diversity_scores = calculate_diversity_metrics(predictions)
    print(f"Diversity scores: {diversity_scores}")
    
    corporate_scores = calculate_corporate_qa_metrics(predictions, references)
    print(f"Corporate Q&A scores: {corporate_scores}")
    
    print("Metrics test completed successfully!")