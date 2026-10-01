"""Практика 10. Три проверки из задания и проверка ошибочного ввода."""
from pathlib import Path
import sys
import socket
import urllib.request
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_predict_function_runs():
    from predict import predict_species
    assert predict_species(5.1,3.5,1.4,0.2) == 'Setosa'


def test_predict_returns_correct_format():
    from predict import predict_species
    result = predict_species(6.0,2.7,5.1,1.6)
    assert isinstance(result, str)
    assert result in {'Setosa','Versicolor','Virginica'}


def test_web_application_launches():
    from app import build_app
    from gradio_client import Client
    # Система выбирает свободный порт, чтобы не мешать уже открытому приложению.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    demo=build_app()
    try:
        demo.launch(server_name='127.0.0.1',server_port=port,share=False,
                    prevent_thread_lock=True,quiet=True)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/config',timeout=15) as response:
            assert response.status == 200
        # Проверяю весь путь: запрос к серверу, функция модели, ответ интерфейса.
        answer=Client(f'http://127.0.0.1:{port}',verbose=False).predict(5.1,3.5,1.4,0.2,api_name='/predict')
        assert 'Setosa' in answer
    finally:
        demo.close()


@pytest.mark.parametrize('values',[[-1,3,1,0.2],[5,3,float('nan'),0.2],[5,None,1,0.2]])
def test_invalid_measurements(values):
    from predict import predict_species
    with pytest.raises(ValueError):
        predict_species(*values)
