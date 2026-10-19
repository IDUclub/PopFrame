PopFrame
========

.. logo-start

.. figure:: https://i.ibb.co/8bD3qr0/DALL-E-2024-05-22-16-49.png
   :alt: PopFrame logo

.. logo-end

|PythonVersion| |Black|

.. description-start

**EN:**
PopFrame is an open source library for modeling regional settlement systems and territory assessment. It provides tools for building a universal information model of a region based on localities, as well as for scenario modeling and analysis.

**RU:**
PopFrame — это open source библиотека для моделирования каркаса расселения и оценки территорий. Она предоставляет инструменты для построения универсальной информационной модели региона на основе населённых пунктов, а также для сценарного моделирования и анализа.

.. description-end

Table of Contents / Содержание
-----------------------------

- `Core features <#core-features>`_
- `Installation <#installation>`_
- `Examples <#examples>`_
- `Project Structure <#project-structure>`_
- `Documentation <#documentation>`_
- `Developing <#developing>`_
- `License <#license>`_
- `Acknowledgments <#acknowledgments>`_
- `Contacts <#contacts>`_

Core features / Основные возможности
-----------------------------------

.. features-start

**EN:**
- Calculate indicators (population, birth rate, mortality) by municipal districts and municipalities.
- Evaluate territories in relation to the settlement system framework.
- Calculate the level of urbanization of the territory.
- Build agglomerations based on the settlement system framework.

**RU:**
- Расчёт показателей (население, рождаемость, смертность) по муниципальным районам и муниципалитетам.
- Оценка территорий относительно каркаса расселения.
- Расчёт уровня урбанизации территории.
- Построение агломераций на основе каркаса расселения.

.. features-end

Installation / Установка
-----------------------

.. installation-start

**EN:**
PopFrame can be installed with ``pip``:

::
   pip install popframe

**RU:**
Установить PopFrame можно через ``pip``:

::
   pip install popframe

.. installation-end

Examples / Примеры
------------------

**EN:**
Below are examples for the main modules of PopFrame.

**RU:**
Ниже приведены примеры для основных модулей PopFrame.

Подготовка Region из OSM через Nominatim и Overpass:
`examples/region_from_osm.ipynb <examples/region_from_osm.ipynb>`_.
Загрузчик находится в ``popframe.preprocessing.data_loader``; итог — один pickle модели.
``RegionDataLoader`` загружает границу, пункты и дорожный граф с повторами и служебным кэшем.
Запускайте ноутбук из ``examples``. Пропуски населения нужно дополнить перед сборкой;
``build_region`` рассчитывает матрицу в минутах, ``save_region`` сохраняет модель
без перезаписи существующего файла. Дополнительные зависимости не нужны.

**1. Region (models):**
```python
from popframe.models.region import Region
region = Region.from_pickle('examples/data/Ленинградская область.pickle')
print(region.towns)
```

**2. PopulationFrame (method):**
```python
from popframe.method.popuation_frame import PopulationFrame
frame = PopulationFrame(region=region)
gdf = frame.build_circle_frame()
gdf.plot()
```

**3. AgglomerationBuilder (method):**
```python
from popframe.method.aglomeration import AgglomerationBuilder
builder = AgglomerationBuilder(region=region)
agglos = builder.get_agglomerations(time=80)
agglos.plot()
```

**4. CityPopulationScorer (method):**
```python
from popframe.method.city_evaluation import CityPopulationScorer
scorer = CityPopulationScorer(gdf_mo, gdf_hex)
results = scorer.run()
```

**5. InfrastructureAnalyzer (method):**
```python
from popframe.method.engineer import InfrastructureAnalyzer
analyzer = InfrastructureAnalyzer(infrastructure_gdf, assessment_areas_gdf)
results = analyzer.get_results()
```

**6. LevelFiller (preprocessing):**
```python
from popframe.preprocessing.level_filler import LevelFiller
filler = LevelFiller(towns=gdf_towns)
leveled = filler.fill_levels()
```

**7. PopulationFiller (preprocessing):**
```python
from popframe.preprocessing.population_filler import PopulationFiller
filler = PopulationFiller(units=gdf_units, towns=gdf_towns, adjacency_matrix=adj_matrix)
filled = filler.fill()
```

**8. Utils (const):**
```python
from popframe.utils.const import *
# Используйте константы для настройки методов
```

See more examples in the `examples/` directory and documentation.
Больше примеров — в папке `examples/` и в документации.

Data for the notebooks is provided under `examples/data`.
- EN: Example notebooks expect inputs from `examples/data` (GeoJSON/Pickle). If you use your own region, update paths accordingly.
- RU: Для запуска ноутбуков используйте данные из `examples/data`. Если работаете со своими данными, поправьте пути в ноутбуках.

Notes on external data access:
- EN: Some examples fetch OpenStreetMap data (e.g., land use). Ensure internet access. If your environment requires VPN (e.g., Forti), connect before running.
- RU: Часть примеров получает данные из OSM. Нужен доступ в интернет (при необходимости подключите VPN, например Forti).

Project Structure / Структура проекта
------------------------------------

**EN:**
- ``popframe`` — library code:
  - ``preprocessing`` — data preprocessing
  - ``models`` — core data models
  - ``method`` — analytical methods
  - ``utils`` — utility functions and constants
- ``examples`` — usage examples
- ``docs`` — documentation sources

**RU:**
- ``popframe`` — код библиотеки:
  - ``preprocessing`` — предобработка данных
  - ``models`` — основные модели данных
  - ``method`` — аналитические методы
  - ``utils`` — утилиты и константы
- ``examples`` — примеры использования
- ``docs`` — исходники документации

Documentation / Документация
----------------------------

**EN:**
Full documentation is available at: https://mvin8.github.io/PopFrame/

**RU:**
Полная документация: https://mvin8.github.io/PopFrame/

Developing / Разработка
----------------------

.. developing-start

**EN:**
To start developing PopFrame:
1. Clone the repository:
   ::
       $ git clone https://github.com/Mvin8/PopFrame
2. (Optional) Create a virtual environment:
   ::
       $ make venv
       $ source .venv/bin/activate
3. Install the library in editable mode with development dependencies:
   ::
       $ make install-dev
4. Install pre-commit hooks:
   ::
       $ pre-commit install
5. Create a new branch:
   ::
       $ git checkout -b develop <new_branch_name>
6. Make changes, update tests and documentation, and run:
   ::
       $ make test
7. Commit and push your changes, then open a Pull Request.

**RU:**
Для начала разработки PopFrame:
1. Клонируйте репозиторий:
   ::
       $ git clone https://github.com/Mvin8/PopFrame
2. (Опционально) создайте виртуальное окружение:
   ::
       $ make venv
       $ source .venv/bin/activate
3. Установите библиотеку в editable-режиме с dev-зависимостями:
   ::
       $ make install-dev
4. Установите pre-commit хуки:
   ::
       $ pre-commit install
5. Создайте новую ветку:
   ::
       $ git checkout -b develop <new_branch_name>
6. Вносите изменения, обновляйте тесты и документацию, запускайте:
   ::
       $ make test
7. Зафиксируйте и отправьте изменения, откройте Pull Request.

.. developing-end

License / Лицензия
------------------

BSD-3-Clause license. See LICENSE file.

Acknowledgments / Благодарности
-------------------------------

.. acknowledgments-start

**EN:**
PopFrame was developed as part of the ITMO University project.

**RU:**
PopFrame разработан в рамках проекта Университета ИТМО.

.. acknowledgments-end

Contacts / Контакты
-------------------

.. contacts-start

- `NCCR <https://actcognitive.org/o-tsentre/kontakty>`__ — National Center for Cognitive Research
- `IDU <https://idu.itmo.ru/en/contacts/contacts.htm>`__ — Institute of Design and Urban Studies
- `Maksim Natykin <https://t.me/Mvin98>`__ — lead software engineer

.. contacts-end



