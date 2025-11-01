


class Action:

    @staticmethod
    def destory(actor, target):

        # Get actor skill parameters
        actor_params = actor.parameters
        target_params = target.parameters

        print(actor_params)
        print(target_params)

    @classmethod
    def available_actions(cls):

        methods = [
            name for name, member in cls.__dict__.items()
            if isinstance(member, staticmethod)
        ]

        return methods


if __name__ == "__main__":
    from Components.subject import Subject
    subject1 = Subject()
    subject2 = Subject()
    from Components.objects import Object
    object1 = Object()

    # Action testing
    Action.destory(subject1, object1)