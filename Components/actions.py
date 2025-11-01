


class Action:

    @staticmethod
    def destory(actor, target):

        # Subject genetics modifiers
        traits = interpret_genetics(actor.genetics)

        base_power = abs(fy)
        power = base_power * traits["strength"] * traits["precision"]
        energy_cost = power * (1 + traits["edge_var"]) / traits["efficiency"]

        actor.energy -= energy_cost
        target.durability -= power * traits["stability"]

        if target.durability <= 0:
            target.destroyed = True
            print(f"Subject: {actor.id} destroyed Object: {target.id}")
        else:
            print(f"Subject: {actor.id} damaged Object: {target.id}. Durability now {target.durability:.2f}")

    @classmethod
    def available_actions(cls):

        methods = [
            name for name, member in cls.__dict__.items()
            if isinstance(member, staticmethod)
        ]

        return methods


if __name__ == "__main__":
    a = Action()
    print(a.available_actions())