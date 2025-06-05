import psycopg2

# Параметры подключения к базе данных
conn_params = {
    'dbname': 'your_database_name',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'your_host',
    'port': 'your_port'
}

try:
    # Подключение к базе данных
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    # Выполнение SQL-запроса
    cursor.execute("SELECT Field1 FROM Table1")

    # Получение всех строк из результата запроса
    rows = cursor.fetchall()

    # Вывод данных на экран
    for row in rows:
        print(row[0])

except psycopg2.Error as e:
    print(f"Ошибка при работе с PostgreSQL: {e}")

finally:
    # Закрытие соединения
    if conn:
        cursor.close()
        conn.close()
        print("Соединение с PostgreSQL закрыто")