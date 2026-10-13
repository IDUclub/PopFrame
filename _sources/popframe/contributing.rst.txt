Contributing / Вклад в проект
============================

**EN:**
We welcome your contributions to PopFrame! Please check the existing issues for bugs or enhancements to work on. If you have an idea for an extension, file a new issue so we can discuss it. Familiarize yourself with the project layout before making major contributions.

**RU:**
Мы приветствуем ваш вклад в PopFrame! Ознакомьтесь с существующими задачами (issues) для поиска багов или улучшений. Если у вас есть идея для расширения, создайте новую задачу для обсуждения. Перед крупными изменениями ознакомьтесь со структурой проекта.

How to contribute / Как внести вклад
------------------------------------

.. include:: ../../../README.rst
   :start-after: .. developing-start
   :end-before: .. developing-end

Before submitting your pull request / Перед отправкой pull request
-----------------------------------------------------------------

**EN:**
- Update the documentation and README if your changes affect them.
- Update or add tests for your code.
- Make sure your code is properly commented and documented.
- If you add dependencies, ensure they are easy to install via pip and support Python 3.

**RU:**
- Обновите документацию и README, если ваши изменения их затрагивают.
- Добавьте или обновите тесты для вашего кода.
- Убедитесь, что код снабжён комментариями и docstring.
- Новые зависимости должны легко устанавливаться через pip и поддерживать Python 3.

Contribute to the documentation / Вклад в документацию
------------------------------------------------------

**EN:**
All documentation is created with Sphinx autodoc. Use .. automodule:: <module_name> for module documentation. Add new files to the toctree in api/index.rst.

**RU:**
Вся документация создаётся с помощью Sphinx autodoc. Для описания модулей используйте .. automodule:: <module_name>. Новые файлы добавляйте в toctree в api/index.rst.

After submitting your pull request / После отправки pull request
---------------------------------------------------------------

**EN:**
- Automated tests and code quality checks will run.
- Address any errors if checks fail.

**RU:**
- Будут запущены автоматические тесты и проверки качества кода.
- Исправьте ошибки, если проверки не пройдены.

Acknowledgements / Благодарности
--------------------------------

This guide is based on the TPOT Framework contribution guide.

Данный гайд основан на руководстве по вкладу в TPOT Framework.