import socket
import time

host = "IP address"
port = 2001


def send_command(command):
    try:
        # Создаем сокет
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Соединение с {host}:{port}")

        # Устанавливаем соединение
        s.connect((host, port))
        print(f"Отправка команды: {command}")

        # Отправляем команду
        s.sendall(command)

        # Добавляем небольшой задержку между отправками команд
        time.sleep(1)

        return True
    except socket.error as e:
        print(f"Ошибка сокета: {e}")
        return False
    finally:
        # Закрываем соединение
        s.close()
        print("Соединение закрыто")


# Первая команда
command = b'\xff\x06\x01\x00\xff'  # Пример отправки команды
result = send_command(command)
print('Команда на установку цвета отправлена: ', result)
