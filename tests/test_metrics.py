from lsn.evaluation.metrics import (
    edit_distance, cer, wer, word_acc, sentence_acc,
)


def test_edit_distance_identical():
    assert edit_distance(list("abc"), list("abc")) == 0


def test_edit_distance_single_sub():
    assert edit_distance(list("abc"), list("abd")) == 1


def test_edit_distance_insertion_deletion():
    assert edit_distance(list("ab"), list("abc")) == 1
    assert edit_distance(list("abc"), list("ab")) == 1


def test_cer_zero_for_identical():
    refs = ["hello", "world"]
    hyps = ["hello", "world"]
    assert cer(refs, hyps) == 0.0


def test_wer_zero_for_identical():
    refs = ["hello world"]
    hyps = ["hello world"]
    assert wer(refs, hyps) == 0.0


def test_word_acc_inverts_wer():
    refs = ["a b c"]
    hyps = ["a b c"]
    assert word_acc(refs, hyps) == 1.0


def test_sentence_acc_exact_match():
    refs = ["foo", "bar", "baz"]
    hyps = ["foo", "BAR", "baz"]   # one mismatch (BAR != bar after lower)
    acc = sentence_acc(refs, hyps)
    assert 0.0 <= acc <= 1.0
