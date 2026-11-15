from app.limits import Window


def test_window_allows_up_to_the_limit_then_refuses():
    w = Window(3, 60)
    assert [w.allow("a", now=t) for t in (0, 1, 2, 3)] == [True, True, True, False]


def test_window_is_per_client():
    w = Window(1, 60)
    assert w.allow("a", now=0) and w.allow("b", now=0) and not w.allow("a", now=1)


def test_window_frees_up_after_its_time():
    w = Window(1, 60)
    assert w.allow("a", now=0) and not w.allow("a", now=59) and w.allow("a", now=61)
