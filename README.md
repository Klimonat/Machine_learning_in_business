# Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy 
API: flask 
Данные: с archive.ics.uci.edu - https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Задача: предсказать, подпишется ли клиент на срочный депозит (поле y). Бинарная классификация

Используемые признаки:

* duration (int)
* nr.employed (float)
* cons.conf.idx (float)
* euribor3m (float)
* pdays (int)

Преобразования признаков: StandardScaler()

Модель: Gradient Boosting Classifier

## Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/Klimonat/Machine_learning_in_business.git
$ git checkout course_project
$ docker build -t klimonat/my_service .
```
Запускаем контейнер
Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)

```
$ docker run -d -p 8180:8180 klimonat/my_service
```
## Переходим на localhost:8180