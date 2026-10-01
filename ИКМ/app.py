"""Практика 9. Простой веб-интерфейс моего проекта Iris."""
import gradio as gr
from predict import prediction_text


def build_app():
    # Порядок полей совпадает с порядком признаков при обучении.
    return gr.Interface(
        fn=prediction_text,
        inputs=[gr.Number(label='Длина чашелистика, см', value=5.1),
                gr.Number(label='Ширина чашелистика, см', value=3.5),
                gr.Number(label='Длина лепестка, см', value=1.4),
                gr.Number(label='Ширина лепестка, см', value=0.2)],
        outputs=gr.Textbox(label='Результат', lines=4),
        title='Определение вида ириса',
        description='Учебный помощник. Введите четыре размера цветка в сантиметрах. Модель выбирает Setosa, Versicolor или Virginica.',
        examples=[[5.1,3.5,1.4,0.2],[6.0,2.7,5.1,1.6]],
        submit_btn='Определить вид', clear_btn='Очистить', flagging_mode='never',
        api_name='predict', analytics_enabled=False)


if __name__ == '__main__':
    # Приложение работает локально. Для запуска обучение повторять не нужно.
    build_app().launch(server_name='127.0.0.1', share=False, inbrowser=True)
