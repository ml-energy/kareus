class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


gen_req_id = Counter()


def get_gen_req_id():
    return gen_req_id.__next__()


def reset_gen_req_id():
    gen_req_id.reset()
