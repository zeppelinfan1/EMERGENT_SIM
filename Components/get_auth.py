import keyring


def get_auth(address: str, username: str) -> (str, str):

    user_n, pass_w = keyring.get_password(service_name=address, username=username).split(".")

    return user_n, pass_w


# TESTING
if __name__ == "__main__":
    # Log username/password combo
    user, pwd = get_auth(address="my_credentials", username="log_db")
    print(user, pwd)
