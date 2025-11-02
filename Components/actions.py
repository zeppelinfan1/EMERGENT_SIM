


class Action:

    @staticmethod
    def destory(actor, target=None, env=None):

        # Get actor skill parameters
        actor_params = actor.parameters
        target_params = target.parameters

        print(actor_params)
        print(target_params)

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
        func(actor=actor, env=env, target=target)



if __name__ == "__main__":
    from Components.subject import Subject
    subject1 = Subject()
    subject2 = Subject()
    from Components.objects import Object
    object1 = Object()

    # Action testing
    a = Action()
    a.act(action="destory", actor=subject1, env=None, target=subject2)