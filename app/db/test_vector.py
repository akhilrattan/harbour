from database import Database


def main():

    db = Database()

    with db.connect() as connection:

        result = connection.execute(
            "SELECT '[1,2,3]'::vector;"
        ).fetchone()

        print("pgvector is working!")
        print(result[0])


if __name__ == "__main__":
    main()