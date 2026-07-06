import random
from collections import defaultdict, Counter


class MarkovChain:
    def __init__(self, order=2, level="word"):
        self.order = order
        self.level = level
        self.transitions = defaultdict(Counter)
        self.starts = []

    def _tokenize(self, text):
        if self.level == "word":
            return text.split()
        return list(text)

    def _join(self, tokens):
        if self.level == "word":
            return " ".join(tokens)
        return "".join(tokens)

    def train(self, text):
        tokens = self._tokenize(text)
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i:i + self.order])
            nxt = tokens[i + self.order]
            self.transitions[state][nxt] += 1
            if tokens[i][:1].isupper():
                self.starts.append(state)
        if not self.starts:
            self.starts = list(self.transitions.keys())

    def generate(self, length=100, seed_state=None):
        state = seed_state if seed_state else random.choice(self.starts)
        output = list(state)
        while len(output) < length:
            counter = self.transitions.get(state)
            if not counter:
                state = random.choice(self.starts)
                output.extend(state)
                continue
            candidates = list(counter.keys())
            weights = list(counter.values())
            nxt = random.choices(candidates, weights=weights, k=1)[0]
            output.append(nxt)
            state = tuple(output[-self.order:])
        return self._join(output[:length])
