


class Action:


    @staticmethod
    def destory(actor, object):

        pass

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