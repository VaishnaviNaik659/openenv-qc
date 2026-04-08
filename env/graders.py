def compute_reward(action, true_label, given_label):
    score = 0.0

    is_correct_gt = (given_label == true_label)

    # 1. Detection (0.3)
    if action.is_correct == is_correct_gt:
        score += 0.3

    # 2. Correction (0.4)
    if not action.is_correct:
        if action.correct_label == true_label:
            score += 0.4
    else:
        if given_label == true_label:
            score += 0.4

    # 3. Confidence Calibration (0.3)
    if action.is_correct == is_correct_gt:
        score += 0.3 * action.confidence
    else:
        score += 0.3 * (1 - action.confidence)

    # 4. Overconfidence penalty
    if action.is_correct != is_correct_gt and action.confidence > 0.9:
        score -= 0.2

    return max(0.0, min(1.0, score))