from dataclasses import dataclass
import random, math



class Action:

    def act(self, x: float, y: float, actor: object, target: object=None):

        genetics = getattr(actor, "genetics")
        print(genetics.density_at(x, y))

    def apply(self, actor: object, target: object, result: dict) -> None:

        pass


if __name__ == "__main__":
    x, y = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)

    from Components.subject import Subject
    subject1 = Subject()
    subject2 = Subject()
    from Components.objects import Object
    object1 = Object()

    # Action testing
    a = Action()
    a.act(x, y, actor=subject1, target=object1)
