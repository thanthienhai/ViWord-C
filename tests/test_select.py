import itertools

from viword.select import exact_select, greedy_select, trim_to_budget


def best_by_enumeration(scores, costs, budget):
    best = 0.0
    for keep in itertools.product([0, 1], repeat=len(scores)):
        if sum(c for c, k in zip(costs, keep) if k) <= budget:
            best = max(best, sum(s for s, k in zip(scores, keep) if k))
    return best


def value(scores, keep):
    return sum(s for s, k in zip(scores, keep) if k)


def test_exact_matches_enumeration():
    scores, costs = [0.9, 0.8, 0.5, 0.4, 0.3, 0.2], [3, 2, 2, 1, 1, 1]
    for budget in range(0, 11):
        keep = exact_select(scores, costs, budget)
        assert sum(c for c, k in zip(costs, keep) if k) <= budget
        assert abs(value(scores, keep) - best_by_enumeration(scores, costs, budget)) < 1e-9


def test_greedy_respects_budget_and_alpha():
    scores, costs = [0.9, 0.6, 0.6], [3, 1, 1]
    # alpha = 0: the highest score first
    assert greedy_select(scores, costs, 3, alpha=0.0, fill=False) == [True, False, False]
    # alpha = 1: score per token, two cheap units beat one expensive unit
    assert greedy_select(scores, costs, 3, alpha=1.0, fill=False) == [False, True, True]


def test_fill_pass_uses_leftover_budget():
    scores, costs = [0.9, 0.8, 0.1], [2, 3, 1]
    assert greedy_select(scores, costs, 3, fill=False) == [True, False, False]
    assert greedy_select(scores, costs, 3, fill=True) == [True, False, True]


def test_priority_goes_first():
    keep = greedy_select([0.9, 0.1], [1, 1], 1, priority=[0, 1])
    assert keep == [False, True]


def test_trim_drops_lowest_scores_until_fit():
    keep = trim_to_budget([True, True, True], [0.5, 0.1, 0.9], 2, count_tokens=lambda k: sum(k))
    assert keep == [True, False, True]
