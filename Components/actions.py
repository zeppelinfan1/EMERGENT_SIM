from dataclasses import dataclass
import random, math

class ActionSpace:

    # Semantic axes
    axes = (
        "create_destroy",  # - destroy, + create
        "consume_produce",  # - consume, + produce
    )

    base_limit: float = 1.0

    # Action axis parameter weights
    """NOTE: Eventually replace these and make them dynamic based on genetics"""
    CREATE_WEIGHTS = {
        "efficiency": 1.2,
        "precision": 0.8,
        "stability": 0.9,
        "edge_mean": 0.4,
    }
    DESTROY_WEIGHTS = {
        "strength": 1.3,
        "aggression": 1.1,
        "edge_var": 0.6,
        "edge_max": 0.5,
    }
    PRODUCE_WEIGHTS = {
        "efficiency": 1.1,
        "precision": 0.7,
        "stability": 0.8,
        "edge_mean": 0.3,
    }
    CONSUME_WEIGHTS = {
        "strength": 1.0,
        "area": 0.9,
        "aggression": 0.4,
    }

    @staticmethod
    def _semantic_axis(params, positive_weights, negative_weights):

        pos = sum(params[k] * w for k, w in positive_weights.items())
        neg = sum(params[k] * w for k, w in negative_weights.items())

        return pos - neg

    def create_space(self):

        self.axis_functions = {
            "create_destroy": lambda params: self._semantic_axis(
                params,
                self.CREATE_WEIGHTS,
                self.DESTROY_WEIGHTS
            ),
            "consume_produce": lambda params: self._semantic_axis(
                params,
                self.PRODUCE_WEIGHTS,
                self.CONSUME_WEIGHTS
            )
        }

    def update_space(self):

        pass



class Action(ActionSpace):

    def act(self, actor: object, target: object=None):

        print(target.parameters)

    def apply(self, actor: object, target: object, result: dict) -> None:

        pass


if __name__ == "__main__":
    from Components.subject import Subject
    subject1 = Subject()
    subject2 = Subject()
    from Components.objects import Object
    object1 = Object()

    # Action testing
    a = Action()
    a.act(actor=subject1, target=object1)
    a.create_space()
    # a.act(action="destory", actor=subject1, env=None, target=subject2)
