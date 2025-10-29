def get_tp_group():
    class DummyTPGroup:
        @property
        def world_size(self):
            return 1
        @property
        def rank(self):
            return 0
    return DummyTPGroup()
