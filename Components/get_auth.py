import keyring


def get_auth(service_name: str, username: str) -> (str, str):

    pwd = keyring.get_password(service_name, username)

    return pwd


# TESTING
if __name__ == "__main__":
    # Log username/password combo
    pwd = get_auth(service_name="mysql", username="dchiappo")
    print(pwd)
