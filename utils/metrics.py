import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


def compute_bleu(reference_captions, predicted_caption):
    """
    Calculate BLEU scores for a single predicted caption.
    :param reference_captions: List of reference captions (each a list of words).
    :param predicted_caption: Predicted caption as a list of words.
    """
    bleu_1 = sentence_bleu(reference_captions, predicted_caption, weights=(1, 0, 0, 0))  # Unigrams
    bleu_2 = sentence_bleu(reference_captions, predicted_caption, weights=(0.5, 0.5, 0, 0))  # Bigrams
    bleu_3 = sentence_bleu(reference_captions, predicted_caption, weights=(0.33, 0.33, 0.33, 0))  # Trigrams
    bleu_4 = sentence_bleu(reference_captions, predicted_caption, weights=(0.25, 0.25, 0.25, 0.25))  # 4-grams
    return bleu_1, bleu_2, bleu_3, bleu_4


def compute_rouge(reference_caption, predicted_caption):
    """
    Compute ROUGE scores for a single predicted caption.
    :param reference_caption: A single reference caption as a string.
    :param predicted_caption: Predicted caption as a string.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_caption, predicted_caption)
    return scores


def compute_meteor(reference_captions, predicted_caption):
    """
    Calculate METEOR score for a single predicted caption.
    :param reference_captions: List of reference captions as strings.
    :param predicted_caption: Predicted caption as a string.
    """
    return meteor_score(reference_captions, predicted_caption)