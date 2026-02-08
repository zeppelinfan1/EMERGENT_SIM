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

        """
        Actor converts energy into structure.
        If target is None, creates a new object in the environment.
        """

        if target is None:
            if env is None:
                raise ValueError("Environment instance required for untargeted create().")
            new_obj = env.spawn_object()
            energy_cost = random.uniform(-5.0, -10.0)
            return {
                "actor": actor,
                "target": new_obj,
                "actor_energy_delta": energy_cost,
                "durability_delta": 0,
                "mode": "untargeted",
            }

        a, t = actor.parameters, target.parameters
        efficiency = a["efficiency"]
        stability = a["stability"]
        precision = a["precision"]

        resistance = (t["edge_mean"] + t["edge_var"]) / 2
        power = (efficiency + stability + precision) / 3
        noise = random.uniform(0.9, 1.1)
        net = (power - resistance) * noise

        if net > 0:
            durability_delta = net * 10
            actor_energy_delta = -durability_delta * 0.4
        else:
            durability_delta = 0
            actor_energy_delta = -abs(net) * 5

        return {
            "actor": actor,
            "target": target,
            "actor_energy_delta": actor_energy_delta,
            "durability_delta": durability_delta,
            "mode": "targeted",
            "power": power,
            "resistance": resistance,
            "net_force": net,
        }

    @staticmethod
    def consume(actor, env, target=None):

        """
        Actor transfers its own energy to strengthen target durability.
        Untargeted: self-repair.
        """

        if target is None:
            target = actor  # self-consume (repair)

        a, t = actor.parameters, target.parameters
        efficiency = a["efficiency"]
        strength = a["strength"]
        stability = t["stability"]

        power = (efficiency + strength) / 2
        resistance = stability
        noise = random.uniform(0.9, 1.1)
        net = (power - resistance) * noise

        if net > 0:
            durability_delta = net * 5
            actor_energy_delta = -durability_delta * 0.6
        else:
            durability_delta = 0
            actor_energy_delta = -abs(net) * 3

        return {
            "actor": actor,
            "target": target,
            "actor_energy_delta": actor_energy_delta,
            "durability_delta": durability_delta,
            "mode": "targeted" if target != actor else "untargeted",
            "power": power,
            "resistance": resistance,
            "net_force": net,
        }

    @staticmethod
    def produce(actor, env, target=None):

        """
        Actor extracts energy from target or environment.
        Untargeted: passive energy absorption.
        """

        if target is None:
            energy_gain = random.uniform(1.0, 3.0)
            return {
                "actor": actor,
                "target": None,
                "actor_energy_delta": energy_gain,
                "durability_delta": 0,
                "mode": "untargeted",
            }

        a, t = actor.parameters, target.parameters
        efficiency = a["efficiency"]
        precision = a["precision"]
        strength = t["strength"]
        stability = t["stability"]

        power = (efficiency + precision) / 2
        resistance = (strength + stability) / 2
        noise = random.uniform(0.9, 1.1)
        net = (power - resistance) * noise

        if net > 0:
            energy_gain = net * 10
            durability_delta = -net * 5
        else:
            energy_gain = 0
            durability_delta = 0

        return {
            "actor": actor,
            "target": target,
            "actor_energy_delta": energy_gain,
            "durability_delta": durability_delta,
            "mode": "targeted",
            "power": power,
            "resistance": resistance,
            "net_force": net,
        }

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

        # Apply effects of result
        self.apply(actor=actor, target=target, result=result)

    def apply(self, actor: object, target: object, result: dict) -> None:

        action = result.get("mode", "unknown").capitalize()
        print(f"\n[Apply] Action mode: {action}")

        # Actor effects
        delta_e = result.get("actor_energy_delta", 0)
        if hasattr(actor, "energy"):
            prev_energy = actor.energy
            actor.energy = max(0, actor.energy + delta_e)
            print(f"Actor {actor.id}: Energy {prev_energy:.2f} → {actor.energy:.2f} (Δ {delta_e:.2f})")

        # Target effects
        delta_d = result.get("durability_delta", 0)
        if hasattr(target, "durability"):
            prev_dura = target.durability
            target.durability = max(0, target.durability + delta_d)
            print(f"Target {target.id}: Durability {prev_dura:.2f} → {target.durability:.2f} (Δ {delta_d:.2f})")

            # Destruction check
            if target.durability <= 0 and not target.is_destroyed:
                target.is_destroyed = True
                print(f"Target {target.id} has been destroyed.")

        # Additional data
        net_force = result.get("net_force")
        if net_force is not None:
            print(f"Net force: {net_force:.3f}")

        power = result.get("power")
        resistance = result.get("resistance")
        if power is not None and resistance is not None:
            print(f"Power: {power:.3f} | Resistance: {resistance:.3f}")

        print("[Apply] Effects applied successfully.")


if __name__ == "__main__":
    from Components.subject import Subject
    subject1 = Subject()
    subject2 = Subject()
    from Components.objects import Object
    object1 = Object()

    # Action testing
    a = Action()
    a.act(action="destory", actor=subject1, env=None, target=subject2)

