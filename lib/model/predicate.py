# Supervision predicates
class Predicate:
    def __init__(self, f):
        self.f = f

    def __call__(self, i, lb, t):
        return self.f(i, lb, t)

    def __and__(self, other):
        return Predicate(lambda *args: self(*args) and other(*args))

    def __or__(self, other):
        return Predicate(lambda *args: self(*args) or other(*args))

    def __invert__(self):
        return Predicate(lambda *args: not self(*args))


# Concrete predicates
class Temporal(Predicate):
    def __init__(self, f):
        super().__init__(lambda i, lb, t: f(t))


class Spatial(Predicate):
    def __init__(self, f):
        super().__init__(lambda i, lb, t: f(i))


class Identity(Predicate):
    def __init__(self, f):
        super().__init__(lambda i, lb, t: f(lb))


# Useful predicates
# Temporal
after = lambda t: Temporal(lambda t_: t_ >= t)
before = lambda t: ~after(t)
period = lambda start, stop: after(start) & before(stop)
periodic = lambda every: Temporal(lambda t_: t_ % every == 0)

# Spatial
select = lambda ixs: Spatial(lambda i: i in ixs)

# Identity
identity = lambda: Identity(lambda l: l)
equal = lambda x: Identity(lambda l: l == x)