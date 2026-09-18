from database import Database


def main():

    db = Database()

    with db.connect() as connection:

        result = connection.execute(
            "SELECT version();"
        ).fetchone()

        print("Connected successfully!")
        print(result[0])


if __name__ == "__main__":
    main()