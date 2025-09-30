os:=windows
db:=
app:=
p:=.
id:=

all:
	echo "Makefile settings"

setup:
	pip install -r requirements.txt
	pip uninstall -y enum34
	make folder-setup-$(os) -i

setup-dev: setup
	pip install -r requirements_dev.txt

run:
	python manage.py runserver

check:
	python manage.py check

shell:
	python manage.py shell

dbshell:
	python manage.py dbshell

m-show:
	python manage.py showmigrations

m-create:
	python manage.py makemigrations $(app)

m-migrate:
	python manage.py migrate $(app)

unit:
	python manage.py test -k $(	)

lint:
	python -m black --check .
	python -m isort --profile black --check --gitignore .

format:
	python -m isort --profile black .
	python -m black .
	python -m pycln . -a

check-format:
	python -m isort --profile black --check .
	python -m black --check .
	python -m pycln --check . -a

prready: lint check-format unit

folder-setup-windows:
	mkdir upload
	mkdir logs
	mkdir logs\uploads
	type nul > logs\access.log
	type nul > logs\errors.log
	type nul > logs\requests.log
	type nul > logs\uploads\crons.log
	type nul > logs\uploads\uploads.log
	type nul > logs\uploads\errors.log

folder-setup-linux:
	mkdir upload logs logs/uploads
	> logs/access.log > logs/errors.log > logs/requests.log
	> logs/uploads/crons.log > logs/uploads/uploads.log > logs/uploads/errors.log

db-setup:
	python manage.py shell -c "from annotations.db.$(db) import main; main()"

annotations-execute:
	python manage.py shell -c "from annotations.db.modifications import $(id); $(id)()"
