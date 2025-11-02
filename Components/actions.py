import random


class Action:

    @staticmethod
    def destory(actor, target=None, env=None):

        """
            Actor attempts to destroy target.
            Power and resistance are determined by genetic parameters.
            """

        if target is None:
            # Untargeted destroy: actor wastes energy into environment
            fatigue = random.uniform(2.0, 5.0)
            actor.energy = max(0, actor.energy - fatigue)
            return {
                "mode": "untargeted",
                "energy_loss": fatigue,
                "effect": "no target"
            }

        # --- Targeted destroy ---
        a = actor.parameters
        t = target.parameters

        # Actor power: strength + aggression + precision
        power = (a["strength"] + a["aggression"] + a["precision"]) / 3

        # Target resistance: stability * (1 + area) - edge_var
        resistance = max(0.1, t["stability"] * (1 + t["area"]) - t["edge_var"])

        # Emergent randomness
        noise = random.uniform(0.9, 1.1)

        # Net destructive force
        net_force = (power - resistance) * noise

        # Apply durability and energy effects
        if net_force > 0:
            damage = net_force * 10  # scale lightly
            target.durability = max(0, target.durability - damage)
            actor.energy = max(0, actor.energy - damage * 0.3)
        else:
            # Recoil or wasted effort
            recoil = abs(net_force) * 5
            actor.energy = max(0, actor.energy - recoil)

        # Check for destruction
        if target.durability <= 0:
            target.is_destroyed = True

        return {
            "mode": "targeted",
            "power": round(power, 3),
            "resistance": round(resistance, 3),
            "net_force": round(net_force, 3),
            "durability_remaining": round(target.durability, 3),
            "actor_energy": round(actor.energy, 3),
            "destroyed": target.is_destroyed
        }

    @staticmethod
    def create(actor, env, target=None):

        pass

    @staticmethod
    def consume(actor, env, target=None):

        pass

    @staticmethod
    def produce(actor, env, target=None):

        pass

    @classmethod
    def available_actions(cls):

        methods = [
            name for name, member in cls.__dict__.items()
            if isinstance(member, staticmethod)
        ]

        return methods

    def act(self, action: str, actor: object, env: object, target: object=None):

        func = getattr(Action, action, None)
        result = func(actor=actor, env=env, target=target)

        print(result)


if __name__ == "__main__":
    from Components.subject import Subject
    subject1 = Subject()
    subject2 = Subject()
    from Components.objects import Object
    object1 = Object()

    # Action testing
    a = Action()
    a.act(action="destory", actor=subject1, env=None, target=subject2)

